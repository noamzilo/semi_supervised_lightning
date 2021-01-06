from Testing.Research.autoencoding.VAEFC_Lightning import train_vae, run_trained_vae
from Testing.Research.classification.LatentClassifierLightning import train_latent_classifier, run_trained_classifier


def train_test_vae():
    vae_trainer = train_vae()
    run_trained_vae(vae_trainer)


def train_test_classifier():
    # using latest vae
    classifier_trainer = train_latent_classifier()
    run_trained_classifier(classifier_trainer)


def train_test_everything():
    train_test_vae()
    train_test_classifier()


if __name__ == "__main__":
    train_test_classifier()
    # train_test_everything()
