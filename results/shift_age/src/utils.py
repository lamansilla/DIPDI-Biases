import pandas as pd
from sklearn.model_selection import StratifiedKFold


def generate_training_distribution(data, option, random_state=None):
    data_male = data[data["a"] == "Male"]
    data_female = data[data["a"] == "Female"]
    total_size = min(len(data_male), len(data_female))

    if option == "male":
        train_data = data_male.sample(n=total_size, replace=False, random_state=random_state)
    elif option == "female":
        train_data = data_female.sample(n=total_size, replace=False, random_state=random_state)
    elif option == "mixed":
        train_data = pd.concat(
            [
                data_male.sample(n=total_size // 2, replace=False, random_state=random_state),
                data_female.sample(n=total_size // 2, replace=False, random_state=random_state),
            ]
        )
    else:
        raise ValueError(f"Invalid option: {option}")

    return train_data.sample(frac=1).reset_index(drop=True)


def generate_training_distribution_group(data, option, group, random_state=None):
    data_male = data[data["a"] == "Male"]
    data_female = data[data["a"] == "Female"]
    total_size = min(len(data_male), len(data_female))

    if option == "male":
        train_data = data_male.sample(n=total_size, replace=False, random_state=random_state)
    elif option == "female":
        train_data = data_female.sample(n=total_size, replace=False, random_state=random_state)
    elif option == "mixed":
        train_data = pd.concat(
            [
                data_male.sample(n=total_size // 2, replace=False, random_state=random_state),
                data_female.sample(n=total_size // 2, replace=False, random_state=random_state),
            ]
        )
    else:
        raise ValueError(f"Invalid option: {option}")

    train_data = train_data.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    for i, (_, group_idx) in enumerate(skf.split(X=train_data.index, y=train_data["a"])):
        train_data.loc[group_idx, "group"] = i

    train_data["group"] = train_data["group"].astype(int)
    train_data_group = train_data[train_data["group"] == group]

    return train_data_group.sample(frac=1).reset_index(drop=True)


def generate_distribution_shift(data, shift, desired_prop, num_samples):
    data_male = data[data["a"] == "Male"]
    data_female = data[data["a"] == "Female"]

    num_samples_per_group = num_samples // 2
    num_samples_younger = int(num_samples * desired_prop) // 2
    num_samples_older = num_samples_per_group - num_samples_younger

    age_limit = 45

    if shift == "both":
        male_sample_younger = data_male[data_male["y"] < age_limit].sample(
            num_samples_younger, replace=True
        )
        male_sample_older = data_male[data_male["y"] >= age_limit].sample(
            num_samples_older, replace=True
        )
        male_sample = pd.concat([male_sample_younger, male_sample_older], ignore_index=True)

        female_sample_younger = data_female[data_female["y"] < age_limit].sample(
            num_samples_younger, replace=True
        )
        female_sample_older = data_female[data_female["y"] >= age_limit].sample(
            num_samples_older, replace=True
        )
        female_sample = pd.concat([female_sample_younger, female_sample_older], ignore_index=True)

    elif shift == "male":
        male_sample_younger = data_male[data_male["y"] < age_limit].sample(
            num_samples_younger, replace=True
        )
        male_sample_older = data_male[data_male["y"] >= age_limit].sample(
            num_samples_older, replace=True
        )
        male_sample = pd.concat([male_sample_younger, male_sample_older], ignore_index=True)

        female_sample = data_female.sample(num_samples_per_group, replace=True)

    elif shift == "female":
        male_sample = data_male.sample(num_samples_per_group, replace=True)

        female_sample_younger = data_female[data_female["y"] < age_limit].sample(
            num_samples_younger, replace=True
        )
        female_sample_older = data_female[data_female["y"] >= age_limit].sample(
            num_samples_older, replace=True
        )
        female_sample = pd.concat([female_sample_younger, female_sample_older], ignore_index=True)
    else:
        raise ValueError(f"Invalid shift: {shift}")

    data_shift = pd.concat([male_sample, female_sample], ignore_index=True)

    return data_shift.sample(frac=1).reset_index(drop=True)
