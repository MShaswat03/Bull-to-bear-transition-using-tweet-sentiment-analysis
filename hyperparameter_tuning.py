import ray
from ray import tune
from train import train_lstm_model


def tune_model(config):
    model, f1 = train_lstm_model(csv_file="results/merged_all_data_enriched.csv",
                                  hidden_size=config["hidden_size"],
                                  num_layers=config["num_layers"],
                                  dropout=config["dropout"],
                                  learning_rate=config["learning_rate"],
                                  num_epochs=100,
                                  batch_size=config["batch_size"])
    tune.report(f1=f1)


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    config = {
        "hidden_size": tune.grid_search([100, 200, 300]),
        "num_layers": tune.grid_search([2, 3, 4]),
        "dropout": tune.grid_search([0.2, 0.3, 0.5]),
        "learning_rate": tune.grid_search([0.0005, 0.001, 0.002]),
        "batch_size": tune.grid_search([32, 64])
    }

    analysis = tune.run(tune_model, config=config, resources_per_trial={"cpu": 2, "gpu": 1})
    best_trial = analysis.get_best_trial(metric="f1", mode="max", scope="all")

    print(f"Best Trial Config: {best_trial.config}")
