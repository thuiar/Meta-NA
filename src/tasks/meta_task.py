import numpy as np
import pickle
from tasks.tools import get_len, noise_mimic, Task

class MetaTask:
    @staticmethod
    def modify_commandline_options(parser):

        parser.add_argument('--task_cache_count', type=int, default=1, help='Number of tasks to sample from cache.')
        parser.add_argument('--min_shots_sup', type=int, default=64, help='Minimum number of support shots.')
        parser.add_argument('--max_shots_sup', type=int, default=512, help='Maximum number of support shots.')
        
        parser.add_argument('--q_sz', type=int, default=128, help='Query set size.')
        parser.add_argument('--br_ratio', type=float, default=0.5, help='Ratio of block or random missing tasks.')
        parser.add_argument('--min_nr', type=float, default=0.0, help='Minimum noise ratio.')
        parser.add_argument('--max_nr', type=float, default=0.9, help='Maximum noise ratio.')
        parser.add_argument('--t_seeds', type=int, default=111, help='Discriminate seeds.')
        parser.add_argument('--inner_step', type=int, default=2, help='Inner step for meta train.')
        parser.add_argument('--inner_lr', type=float, default=0.001, help='Inner learning rate.')
        parser.add_argument('--inner_batch_size', type=int, default=256, help='Inner batch size.')

        parser.add_argument('--outer_step', type=int, default=400, help='Outer step for meta train.')
        parser.add_argument('--test_interval', type=int, default=50, help='Outer step for meta val.')
        
        return parser

    def __init__(self, opt) -> None:
        self.task_cache_count = opt.task_cache_count # Sample n tasks in cache once.
        # Task construction hyper-parameters.
        self.max_shots_sup = opt.max_shots_sup
        self.min_shots_sup = opt.min_shots_sup
        self.q_sz = opt.q_sz
        
        self.br_ratio = opt.br_ratio
        self.min_nr, self.max_nr = opt.min_nr, opt.max_nr
        # discriminate / random seeds.
        self.discriminate = True if opt.t_seeds else False
        self.t_seeds = opt.t_seeds        

        # Load data from pkl.
        with open(f'data/{opt.dataset}_unaligned.pkl', 'rb') as f:
            data = pickle.load(f)
        
        self.tr_text, self.tr_audio, self.tr_vision = data['train']['text_bert'], data['train']['audio'], data['train']['vision']
        self.vl_text, self.vl_audio, self.vl_vision = data['valid']['text_bert'], data['valid']['audio'], data['valid']['vision']
        self.ts_text, self.ts_audio, self.ts_vision = data['test']['text_bert'], data['test']['audio'], data['test']['vision']
        
        self.tr_t_len, self.tr_a_len, self.tr_v_len = get_len(self.tr_text[:, 1, :]), \
             np.array(data['train']['audio_lengths']), np.array(data['train']['vision_lengths'])
        self.vl_t_len, self.vl_a_len, self.vl_v_len = get_len(self.vl_text[:, 1, :]), \
            np.array(data['valid']['audio_lengths']), np.array(data['valid']['vision_lengths'])
        self.ts_t_len, self.ts_a_len, self.ts_v_len = get_len(self.ts_text[:, 1, :]), \
            np.array(data['test']['audio_lengths']), np.array(data['test']['vision_lengths'])

        self.tr_label, self.vl_label, self.ts_label = data['train']['regression_labels'], data['valid']['regression_labels'], data['test']['regression_labels']
        self.tr_id, self.vl_id, self.ts_id = np.array(data['train']['id']), np.array(data['valid']['id']), np.array(data['test']['id'])
        # Task Sampling.
        np.random.seed(opt.t_seeds)
        self.n_count = {"train": len(self.tr_text), "valid": len(self.vl_text), "test": len(self.ts_text)}

        self.datasets_cache = {
            mode: self.load_task_cache(mode)
                for mode in ["meta_train", "meta_valid", "meta_test"]
        }

    def load_task_cache(self, mode):
        """ Pre-Load self.task_cache_count tasks for each mode.
        """
        task_cache = []

        for i in range(self.task_cache_count):
            setsz = np.random.randint(self.min_shots_sup, self.max_shots_sup + 1)
            n_t = 'rand' if np.random.uniform(0,1,1) < 0.5 else 'block'
            n_r = np.round(np.random.uniform(self.min_nr, self.max_nr, 3), decimals=2)

            n = { m: (n_t, n_r[j]) for j,m in enumerate(['text', 'audio', 'vision'])}

            if mode == 'meta_train':
                id_s = np.random.choice(self.n_count['train'], setsz, replace=False)
                id_q = np.random.choice(self.n_count['valid'], self.q_sz, replace=False)

                text_sup, text_qry = self.tr_text[id_s], self.vl_text[id_q]
                audio_sup, audio_qry = self.tr_audio[id_s], self.vl_audio[id_q]
                vision_sup, vision_qry = self.tr_vision[id_s], self.vl_vision[id_q]
                tl_sup, tl_qry = self.tr_t_len[id_s], self.vl_t_len[id_q]
                al_sup, al_qry = self.tr_a_len[id_s], self.vl_a_len[id_q]
                vl_sup, vl_qry = self.tr_v_len[id_s], self.vl_v_len[id_q]
                sup_id, qry_id = self.tr_id[id_s], self.vl_id[id_q]
                sup_y, qry_y = self.tr_label[id_s], self.vl_label[id_q]
            elif mode == 'meta_valid':
                id_s = np.random.choice(self.n_count['train'], setsz, replace=False)
                id_q = np.random.choice(self.n_count['valid'], self.q_sz, replace=False)

                text_sup, text_qry = self.tr_text[id_s], self.vl_text[id_q[setsz:]]
                audio_sup, audio_qry = self.tr_audio[id_s], self.vl_audio[id_q[setsz:]]
                vision_sup, vision_qry = self.tr_vision[id_s], self.vl_vision[id_q[setsz:]]
                tl_sup, tl_qry = self.tr_t_len[id_s], self.vl_t_len[id_q[setsz:]]
                al_sup, al_qry = self.tr_a_len[id_s], self.vl_a_len[id_q[setsz:]]
                vl_sup, vl_qry = self.tr_v_len[id_s], self.vl_v_len[id_q[setsz:]]
                sup_id, qry_id = self.tr_id[id_s], self.vl_id[id_q]
                sup_y, qry_y = self.tr_label[id_s], self.vl_label[id_q[setsz:]]
            elif mode == 'meta_test':
                id_s = np.random.choice(self.n_count['train'], setsz, replace=False)
                id_q = np.random.choice(self.n_count['valid'], self.q_sz, replace=False)

                text_sup, text_qry = self.tr_text[id_s[:setsz]], self.ts_text[id_q[setsz:]]
                audio_sup, audio_qry = self.tr_audio[id_s[:setsz]], self.ts_audio[id_q[setsz:]]
                vision_sup, vision_qry = self.tr_vision[id_s[:setsz]], self.ts_vision[id_q[setsz:]]
                tl_sup, tl_qry = self.tr_t_len[id_s[:setsz]], self.ts_t_len[id_q[setsz:]]
                al_sup, al_qry = self.tr_a_len[id_s[:setsz]], self.ts_a_len[id_q[setsz:]]
                vl_sup, vl_qry = self.tr_v_len[id_s[:setsz]], self.ts_v_len[id_q[setsz:]]
                sup_id, qry_id = self.tr_id[id_s], self.vl_id[id_q]
                sup_y, qry_y = self.tr_label[id_s[:setsz]], self.ts_label[id_q[setsz:]]
            
            sup_x = noise_mimic({
                'text': text_sup, 'audio': audio_sup, 'vision': vision_sup,'id':sup_id,
                'text_len': tl_sup, 'audio_len': al_sup, 'vision_len': vl_sup
            }, noise=n)

            qry_x = noise_mimic({
                'text': text_qry, 'audio': audio_qry, 'vision': vision_qry,'id':qry_id,
                'text_len': tl_qry, 'audio_len': al_qry, 'vision_len': vl_qry
            }, noise=n)

            task_cache.append(Task(
                meta_info={'sup': {'ins': setsz, 'n': n}, 'qry':{'ins': self.q_sz, 'n': n}},
                sup_x=sup_x, sup_y=sup_y, qry_x=qry_x, qry_y=qry_y
            ))

        return task_cache

    def next(self, mode='meta_train'):
        """Load next batch of tasks."""
        self.datasets_cache[mode] = self.load_task_cache(mode)
