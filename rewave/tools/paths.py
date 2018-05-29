def get_model_path(window_length, predictor_type, use_batch_norm):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'
    return 'weights/stock/{}/window_{}/{}'.format(predictor_type, window_length, batch_norm_str)


def get_root_model_dir_path(window_length, predictor_type, use_batch_norm):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'
    return 'weights/root_model/{}/window_{}/{}'.format(predictor_type, window_length, batch_norm_str)

def get_root_model_path(window_length, predictor_type, use_batch_norm):
    return get_root_model_dir_path(window_length, predictor_type, use_batch_norm) + '/root_model.h5'


def get_result_path(window_length, predictor_type, use_batch_norm):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'
    return 'results/stock/{}/window_{}/{}/'.format(predictor_type, window_length, batch_norm_str)