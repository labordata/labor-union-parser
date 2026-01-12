"""Training script for the three-stage extraction pipeline."""

import click
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from labor_union_parser.conf import RANDOM_SEED
from training.evaluate import evaluate_pipeline, print_results
from training.train_affiliation import train_affiliation_classifier
from training.train_common import DATA_DIR, DEVICE, WEIGHTS_DIR
from training.train_designation import train_designation_extractor
from training.train_union import train_union_detector


def stage_callback(ctx, param, value):
    """Convert stage choice into a training callable."""
    if value is None:
        return None

    if value == "1":
        return lambda: train_union_detector(train_indices=None)

    if value == "2":
        return lambda: train_affiliation_classifier(train_indices=None)

    if value == "3":
        aff_model_path = WEIGHTS_DIR / "contrastive_aff_model.pt"
        if not aff_model_path.exists():
            raise click.BadParameter(
                "No affiliation model found. Run stage 2 first.",
                param_hint="'--stage'",
            )
        state = torch.load(aff_model_path, map_location=DEVICE, weights_only=False)
        aff_list = state["aff_list"]
        print(
            f"\nLoaded aff_list with {len(aff_list)} affiliations from existing model"
        )
        return lambda: train_designation_extractor(aff_list, train_indices=None)

    return None


def create_held_out_test_set():
    """Create a held-out test set before any training and save to CSV."""
    print("\n" + "=" * 60)
    print("CREATING HELD-OUT TEST SET")
    print("=" * 60)

    # Load all data
    df = pd.read_csv(DATA_DIR / "labeled_data.csv")
    known_df = df[~df["aff_abbr"].isin(["UNK"])]

    # Filter out classes with < 2 members (can't stratify)
    aff_counts = known_df["aff_abbr"].value_counts()
    valid_affs = aff_counts[aff_counts >= 2].index
    stratifiable_df = known_df[known_df["aff_abbr"].isin(valid_affs)]
    unstratifiable_df = known_df[~known_df["aff_abbr"].isin(valid_affs)]

    # Stratified split on stratifiable data
    train_strat, test_strat = train_test_split(
        stratifiable_df,
        test_size=0.05,
        random_state=RANDOM_SEED,
        stratify=stratifiable_df["aff_abbr"],
    )

    # Add unstratifiable to train (too rare for test)
    train_df = pd.concat([train_strat, unstratifiable_df])
    test_df = test_strat

    print(f"  Total known examples: {len(known_df)}")
    print(f"  Training pool: {len(train_df)}")
    print(f"  Held-out test: {len(test_df)}")

    # Update split column in the dataframe and save
    df["split"] = "train"  # Default
    df.loc[test_df.index, "split"] = "test"
    df.loc[df["aff_abbr"] == "UNK", "split"] = "exclude"
    df.to_csv(DATA_DIR / "labeled_data.csv", index=False)
    print(f"  Saved split to {DATA_DIR / 'labeled_data.csv'}")

    # Return indices for use in training
    train_indices = set(train_df.index.tolist())

    return train_indices, test_df


@click.command()
@click.option(
    "--stage",
    type=click.Choice(["1", "2", "3"]),
    callback=stage_callback,
    expose_value=True,
    is_eager=True,
    help="Train only specific stage (1=union, 2=affiliation, 3=designation)",
)
def main(stage):
    """Train the three-stage extraction pipeline.

    \b
    Stage 1: Union Detection (contrastive)
    Stage 2: Affiliation Classification (contrastive centroids)
    Stage 3: Designation Extraction (pointer network)
    """
    print("=" * 60)
    print("CONTRASTIVE EXTRACTION PIPELINE TRAINING")
    print("=" * 60)
    print(f"\nDevice: {DEVICE}")
    print(f"Weights directory: {WEIGHTS_DIR}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if stage is not None:
        stage()
        return

    # Full pipeline training
    # Create held-out test set BEFORE any training
    train_indices, test_df = create_held_out_test_set()

    # Stage 1
    train_union_detector(train_indices)

    # Stage 2
    _, aff_list, _ = train_affiliation_classifier(train_indices)

    # Stage 3
    train_designation_extractor(aff_list, train_indices)

    # Final evaluation on held-out test set
    print("\n" + "=" * 60)
    print("EVALUATING ON HELD-OUT TEST SET")
    print("=" * 60)
    metrics = evaluate_pipeline(test_df, save_errors=False)
    print_results(metrics)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nAll weights saved to {WEIGHTS_DIR}")
    print("\nTo use the trained models:")
    print("  from labor_union_parser import Extractor")
    print("  extractor = Extractor()")
    print("  result = extractor.extract('SEIU Local 1199')")


if __name__ == "__main__":
    main()
