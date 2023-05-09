import numpy as np

import pickle
from tasks.tools import get_len, noise_mimic, TestTask

class EvalTask:

    @staticmethod
    def modify_commandline_options(parser):
        # n_shots
        # parser.add_argument('--n_shots', type=int, default=1284, help='Number of shots.')
        # model learning parameters.
        
        return parser

    def __init__(self, opt) -> None:
        # self.n_shots = opt.n_shots

        self.task_cache = []

        # Load data from pkl.
        with open(f'data/{opt.dataset}_unaligned.pkl', 'rb') as f:
            data = pickle.load(f)

        self.tr_text, self.tr_audio, self.tr_vision = data['train']['text_bert'], data['train']['audio'], data['train']['vision']
        self.vl_text, self.vl_audio, self.vl_vision = data['valid']['text_bert'], data['valid']['audio'], data['valid']['vision']
        self.ts_text, self.ts_audio, self.ts_vision = data['test']['text_bert'], data['test']['audio'], data['test']['vision']

        self.tr_t_len, self.tr_a_len, self.tr_v_len = get_len(self.tr_text[:, 1, :]), \
             data['train']['audio_lengths'], data['train']['vision_lengths']
        self.vl_t_len, self.vl_a_len, self.vl_v_len = get_len(self.vl_text[:, 1, :]), \
            data['valid']['audio_lengths'], data['valid']['vision_lengths']
        self.ts_t_len, self.ts_a_len, self.ts_v_len = get_len(self.ts_text[:, 1, :]), \
            data['test']['audio_lengths'], data['test']['vision_lengths']

        self.tr_label = data['train']['regression_labels']
        self.vl_label = data['valid']['regression_labels']
        self.ts_label = data['test']['regression_labels']

        for n_t in ['block', 'rand']:
            for n_r in np.arange(0.0, 1.1, 0.1):
                n = {'text': (n_t, n_r), 'audio': (n_t, n_r), 'vision': (n_t, n_r)}

                train_x = noise_mimic({
                    'text': self.tr_text, 'audio': self.tr_audio, 'vision': self.tr_vision,
                    'text_len': self.tr_t_len, 'audio_len': self.tr_a_len, 'vision_len': self.tr_v_len
                }, noise=n)

                val_x = noise_mimic({
                    'text': self.vl_text, 'audio': self.vl_audio, 'vision': self.vl_vision,
                    'text_len': self.vl_t_len, 'audio_len': self.vl_a_len, 'vision_len': self.vl_v_len
                }, noise=n)

                test_x = noise_mimic({
                    'text': self.ts_text, 'audio': self.ts_audio, 'vision': self.ts_vision,
                    'text_len': self.ts_t_len, 'audio_len': self.ts_a_len, 'vision_len': self.ts_v_len
                }, noise=n)

                self.task_cache.append(TestTask(
                    meta_info={'train': {'ins': len(self.tr_label), 'n': n}, 'val':{'ins': len(self.vl_label), 'n': n}, 'test':{'ins': len(self.ts_label), 'n': n}},
                    train_x=train_x, train_y=self.tr_label, val_x=val_x, val_y=self.vl_label,test_x=test_x, test_y=self.ts_label
                ))

        for m in ['text', 'audio', 'vision']:
            n = {
                'text': ('block', 1.0 if m == 'text' else 0.0), 
                'audio': ('block', 1.0 if m == 'audio' else 0.0), 
                'vision': ('block', 1.0 if m == 'vision' else 0.0)
            }

            train_x = noise_mimic({
                'text': self.tr_text, 'audio': self.tr_audio, 'vision': self.tr_vision,
                'text_len': self.tr_t_len, 'audio_len': self.tr_a_len, 'vision_len': self.tr_v_len
            }, noise=n)

            val_x = noise_mimic({
                'text': self.vl_text, 'audio': self.vl_audio, 'vision': self.vl_vision,
                'text_len': self.vl_t_len, 'audio_len': self.vl_a_len, 'vision_len': self.vl_v_len
            }, noise=n)

            test_x = noise_mimic({
                'text': self.ts_text, 'audio': self.ts_audio, 'vision': self.ts_vision,
                'text_len': self.ts_t_len, 'audio_len': self.ts_a_len, 'vision_len': self.ts_v_len
            }, noise=n)

            self.task_cache.append(TestTask(
                meta_info={'train': {'ins': len(self.tr_label), 'n': f'{m}_modality_missing'}, 'val':{'ins': len(self.vl_label), 'n': f'{m}_modality_missing'}, 'test':{'ins': len(self.ts_label), 'n': f'{m}_modality_missing'}},
                train_x=train_x, train_y=self.tr_label, val_x=val_x, val_y=self.vl_label ,test_x=test_x, test_y=self.ts_label
            ))
