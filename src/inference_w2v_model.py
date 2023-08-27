from gensim.models import KeyedVectors
import pickle, os

models_folder_path = "./models"


embeddings_path = os.path.join(models_folder_path, "embeddings.wv")
train_loss_path = os.path.join(models_folder_path, "trainloss_per_epoch.pkl")

model = KeyedVectors.load(embeddings_path)

# Load the training loss per epoch data
with open(train_loss_path, "rb") as f:
    train_loss_per_epoch = pickle.load(f)


target_word = 3056
similar_words = model.most_similar(target_word, topn=5)
print(f"Products similar to '{target_word}':")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")



