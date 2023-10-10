import pickle

if __name__ == "__main__":
    with open("./pretrained/checkpoint_vit_base_tomo.pkl", "rb") as f:
        ckpt = pickle.load(f)

    print(ckpt)