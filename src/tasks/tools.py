import numpy as np

class Task:
    def __init__(self, meta_info, sup_x, sup_y, qry_x, qry_y) -> None:
        self.meta_info = meta_info
        
        self.sup_x = sup_x
        self.sup_y = sup_y
        self.qry_x = qry_x
        self.qry_y = qry_y

    def __repr__(self) -> str:
        return f"T-Sup: {self.meta_info['sup']}, T-Qry: {self.meta_info['qry']}"

    def get_task_d(self):
        return self.sup_x, self.sup_y, self.qry_x, self.qry_y
    
class TestTask:
    def __init__(self, meta_info, train_x, train_y, val_x, val_y, test_x, test_y) -> None:
        self.meta_info = meta_info
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.test_x = test_x
        self.test_y = test_y

    def __repr__(self) -> str:
        return f"Train: {self.meta_info['train']}, Val: {self.meta_info['val']}, Test: {self.meta_info['test']}"

    def get_task_d(self):
        return self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y

def get_len(mask):
    zeros = np.zeros((mask.shape[0], 1))
    mask_ = np.concatenate((mask, zeros), axis=1)
    return np.argmin(mask_, axis=1)
    
def noise_mimic(x, noise):
    """
    x: Instances List to be corrupted. 
        {'text': [n, 3, t], 'audio': [n, t, d], 'vision': [n, t, d], 'text_len': [n], 'audio_len': [n], 'vision_len': [n]}
    noise: Specific noise type and degree for each modality.
        Dict[modality, Tuple[noise_type, noise_degree]]
    """
    for m in ['text', 'audio', 'vision']:
        n_t, n_r = noise[m]
        seq, l = x[m], np.array(x[f'{m}_len'])

        mask = (np.arange(seq.shape[2] if m == 'text' else seq.shape[1])[None, :] < l[:, None]).astype(int)
        if n_t == 'rand':
            missing_mask = (np.random.uniform(size=mask.shape) > n_r) * mask
        elif n_t == 'block':
            missing_block_len = np.around(l * n_r).astype(np.int32)
            missing_mask = mask.copy()
            for i, instance in enumerate(missing_mask):
                s = l[i] - missing_block_len[i]
                start_p = np.random.randint(s) if s > 0 else 0
                missing_mask[i, start_p:start_p+missing_block_len[i]] = 0
        else:
            raise(f"Noise type {n_t} not supported.")

        if m == 'text':
            for i, instance in enumerate(missing_mask):
                instance[0] = instance[l[i] - 1] = 1 # Keep the first and last token.
            m_g = missing_mask * seq[:,0,:] + (100 * np.ones_like(seq[:,0,:])) * (mask - missing_mask) # UNK token: 100.
            m_g = np.concatenate((np.expand_dims(m_g, 1), seq[:,1:,:]), axis=1)
        else:
            m_g = np.expand_dims(missing_mask, axis=2) * seq
        
        x[m] = m_g
    
    return x

def creatNoiseClean(data,data_type,q_sz):
    """
        Perform the same operation to ensure data consistency between clean and noise
    """
    if data_type == 'clean':
        return {
            'audio':data.tr_audio[:q_sz],
            'label':data.tr_label[:q_sz],
            'text':data.tr_text[:q_sz],
            'vision':data.tr_vision[:q_sz],
            'text_len': data.tr_t_len[:q_sz],
            'audio_len': data.tr_a_len[:q_sz],
            'vision_len': data.tr_v_len[:q_sz],
        }
    elif data_type == 'noise':
        return {
            'audio':data['audio'][:q_sz],
            'text':data['text'][:q_sz],
            'vision':data['vision'][:q_sz]
        }