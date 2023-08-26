import argparse

# from ray.tune import Analysis
from utils.data_handling import load_ecomm, train_test_split
from src.modeling import train_w2v, RecallAtKLogger, LossLogger
from src.metrics import recall_at_k, mrr_at_k
from utils.file_handling import absolute_filename, pickle_save


# load data
sessions = load_ecomm()
print(len(sessions))
train, test, valid = train_test_split(sessions, test_size=1000)

# determine word2vec parameters to train with

w2v_params = {
        "min_count": 1,
        "workers": 10,
        "sg": 1,
    }

ratk_logger = RecallAtKLogger(valid, k=5, save_model=True)
loss_logger = LossLogger()
embeddings = train_w2v(train, w2v_params, [ratk_logger, loss_logger])

# Save results
pickle_save(ratk_logger.recall_scores, absolute_filename('models/', f"recall@k_per_epoch.pkl"))
pickle_save(loss_logger.training_loss, absolute_filename('models/', f"trainloss_per_epoch.pkl"))

# Save trained embeddings
embeddings.save(absolute_filename('models/', f"embeddings.wv"))


print(recall_at_k(test, embeddings, k=5))
print(mrr_at_k(test, embeddings, k=5))

