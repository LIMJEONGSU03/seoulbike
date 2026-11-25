import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATE_FORMAT = "%d/%m/%Y"
DEFAULT_TARGET = datetime(2017, 12, 1, 10)
WEATHER_COLUMNS = (
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)",
)
WEATHER_PREFIXES = ("Temperature(", "Dew point temperature(")
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def _column_name(fieldnames: list[str], name: str) -> str:
    if name in fieldnames:
        return name
    for prefix in WEATHER_PREFIXES:
        if name.startswith(prefix):
            return next(field for field in fieldnames if field.startswith(prefix))
    raise ValueError(f"Required CSV column not found: {name}")


def _read_known_rows(csv_path: Path, latest_known_time: datetime) -> list[dict[str, str]]:
    known_rows = []
    previous_time = None

    with csv_path.open(encoding="cp1252", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        columns = {
            name: _column_name(reader.fieldnames, name)
            for name in WEATHER_COLUMNS
        }

        for row in reader:
            observation_time = datetime.strptime(
                f"{row['Date']} {int(row['Hour']):02d}", f"{DATE_FORMAT} %H"
            )
            if previous_time is not None and observation_time <= previous_time:
                raise ValueError("CSV observations must be in strictly increasing time order.")
            previous_time = observation_time

            if observation_time > latest_known_time:
                break
            row["_time"] = observation_time.isoformat()
            row["_weather_columns"] = columns
            known_rows.append(row)
            if observation_time == latest_known_time:
                break

    if not known_rows or datetime.fromisoformat(known_rows[-1]["_time"]) != latest_known_time:
        raise ValueError(
            f"No observation found at {latest_known_time:%Y-%m-%d %H:00}."
        )
    return known_rows


def _features(
    origin: datetime, rows_by_time: dict[datetime, dict[str, str]], target_hour: int
) -> list[float]:
    origin_row = rows_by_time[origin]
    feature_values = []
    for lag in range(3):
        source_time = origin - timedelta(hours=lag)
        feature_values.append(float(rows_by_time[source_time]["Rented Bike Count"]))

    columns = origin_row["_weather_columns"]
    feature_values.extend(float(origin_row[columns[name]]) for name in WEATHER_COLUMNS)

    target_time = origin + timedelta(hours=1)
    feature_values.extend(
        [
            float(target_hour),
            float(target_time.weekday()),
            float(origin_row["Holiday"] != "No Holiday"),
            float(origin_row["Functioning Day"] == "Yes"),
            float(origin_row["Seasons"] == "Winter"),
        ]
    )
    return feature_values


def _fit_model(
    known_rows: list[dict[str, str]], target: datetime
) -> tuple[object, dict[datetime, dict[str, str]], int, float, float, float]:
    rows_by_time = {
        datetime.fromisoformat(row["_time"]): row for row in known_rows
    }

    training_features = []
    training_targets = []
    persistence_predictions = []
    for target_time, target_row in rows_by_time.items():
        if target_time.date() != target.date() or target_time.hour > target.hour - 1:
            continue
        origin = target_time - timedelta(hours=1)
        required_times = [origin - timedelta(hours=lag) for lag in range(3)]
        if not all(time in rows_by_time for time in required_times):
            continue
        training_features.append(_features(origin, rows_by_time, target_time.hour))
        training_targets.append(float(target_row["Rented Bike Count"]))
        persistence_predictions.append(
            float(rows_by_time[origin]["Rented Bike Count"])
        )

    if len(training_targets) < 4:
        raise ValueError(
            "At least four pre-target training examples are needed for walk-forward validation."
        )

    validation_errors = {}
    for alpha in RIDGE_ALPHAS:
        errors = []
        for validation_index in range(3, len(training_targets)):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(
                training_features[:validation_index],
                training_targets[:validation_index],
            )
            predicted = model.predict(
                [training_features[validation_index]]
            )[0]
            errors.append(abs(training_targets[validation_index] - predicted))
        validation_errors[alpha] = sum(errors) / len(errors)

    best_alpha = min(validation_errors, key=validation_errors.get)
    model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
    model.fit(training_features, training_targets)
    persistence_mae = sum(
        abs(training_targets[index] - persistence_predictions[index])
        for index in range(3, len(training_targets))
    ) / (len(training_targets) - 3)
    return (
        model,
        rows_by_time,
        len(training_targets),
        best_alpha,
        persistence_mae,
        validation_errors[best_alpha],
    )


def make_predictions(csv_path: Path, target: datetime) -> tuple[int, int, int, float, float, float]:
    latest_known_time = target - timedelta(hours=1)
    known_rows = _read_known_rows(csv_path, latest_known_time)
    model, rows_by_time, training_count, best_alpha, persistence_mae, linear_mae = (
        _fit_model(known_rows, target)
    )
    linear_prediction = int(
        round(
            max(
                0.0,
                model.predict(
                    [_features(latest_known_time, rows_by_time, target.hour)]
                )[0],
            )
        )
    )
    persistence_prediction = int(rows_by_time[latest_known_time]["Rented Bike Count"])
    return (
        persistence_prediction,
        linear_prediction,
        training_count,
        best_alpha,
        persistence_mae,
        linear_mae,
    )


def _read_evaluation_rows(
    csv_path: Path, target: datetime
) -> list[dict[str, str]]:
    start_time = target - timedelta(hours=1)
    end_time = target.replace(hour=23)
    evaluation_rows = []
    previous_time = None

    with csv_path.open(encoding="cp1252", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        columns = {
            name: _column_name(reader.fieldnames, name)
            for name in WEATHER_COLUMNS
        }

        for row in reader:
            observation_time = datetime.strptime(
                f"{row['Date']} {int(row['Hour']):02d}", f"{DATE_FORMAT} %H"
            )
            if previous_time is not None and observation_time <= previous_time:
                raise ValueError("CSV observations must be in strictly increasing time order.")
            previous_time = observation_time

            if observation_time < start_time:
                continue
            if observation_time > end_time:
                break
            row["_time"] = observation_time.isoformat()
            row["_weather_columns"] = columns
            evaluation_rows.append(row)
            if observation_time == end_time:
                break

    expected_times = [
        target - timedelta(hours=1) + timedelta(hours=hour)
        for hour in range(15)
    ]
    actual_times = [datetime.fromisoformat(row["_time"]) for row in evaluation_rows]
    if actual_times != expected_times:
        raise ValueError("Expected complete hourly observations from 09:00 through 23:00.")
    return evaluation_rows


def make_daily_plot(
    csv_path: Path, target: datetime, output_path: Path
) -> tuple[float, float]:
    import matplotlib.pyplot as plt

    latest_known = target - timedelta(hours=1)
    known_rows = _read_known_rows(csv_path, latest_known)
    model, known_by_time, _, _, _, _ = _fit_model(known_rows, target)

    evaluation_rows = _read_evaluation_rows(csv_path, target)
    rows_by_time = {
        **known_by_time,
        **{
            datetime.fromisoformat(row["_time"]): row
            for row in evaluation_rows
        },
    }

    forecast_times = [
        target + timedelta(hours=offset) for offset in range(24 - target.hour)
    ]
    actual = []
    baseline = []
    linear = []
    for forecast_time in forecast_times:
        origin = forecast_time - timedelta(hours=1)
        baseline.append(int(rows_by_time[origin]["Rented Bike Count"]))
        prediction = model.predict(
            [_features(origin, rows_by_time, forecast_time.hour)]
        )[0]
        linear.append(int(round(max(0.0, prediction))))
        actual.append(int(rows_by_time[forecast_time]["Rented Bike Count"]))

    baseline_mae = sum(
        abs(actual_value - predicted_value)
        for actual_value, predicted_value in zip(actual, baseline)
    ) / len(actual)
    linear_mae = sum(
        abs(actual_value - predicted_value)
        for actual_value, predicted_value in zip(actual, linear)
    ) / len(actual)

    plot_times = [
        latest_known + timedelta(hours=offset) for offset in range(15)
    ]
    actual_by_time = {
        datetime.fromisoformat(row["_time"]): int(row["Rented Bike Count"])
        for row in evaluation_rows
    }
    plt.figure(figsize=(12, 6))
    plt.plot(
        plot_times,
        [actual_by_time[time] for time in plot_times],
        color="black",
        label="Actual",
    )
    plt.plot(
        forecast_times,
        baseline,
        color="gray",
        linestyle="--",
        label="Persistence baseline",
    )
    plt.plot(forecast_times, linear, color="blue", label="Linear regression (Ridge)")
    plt.xlabel("Current time (KST)")
    plt.ylabel("Rented bike count")
    plt.title("Seoul bike rentals: 2017-12-01, 09:00-23:00")
    plt.xticks(plot_times, [time.strftime("%H:%M") for time in plot_times], rotation=45)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return baseline_mae, linear_mae


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare last-hour persistence with a linear regression forecast."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).with_name("SeoulBikeData.csv"),
        help="Path to SeoulBikeData.csv",
    )
    parser.add_argument(
        "--date",
        default=DEFAULT_TARGET.strftime(DATE_FORMAT),
        help="Target date in DD/MM/YYYY format",
    )
    parser.add_argument("--hour", type=int, default=DEFAULT_TARGET.hour)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("SeoulBike.csv"),
        help="Path for the forecast comparison CSV",
    )
    parser.add_argument(
        "--plot-day",
        action="store_true",
        help="Plot actuals from the hour before the target through 23:00 and forecast the rest",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path(__file__).with_name("SeoulBikeData.png"),
        help="Path for the daily forecast plot",
    )
    args = parser.parse_args()

    target = datetime.strptime(f"{args.date} {args.hour:02d}", f"{DATE_FORMAT} %H")
    if args.plot_day:
        baseline_mae, linear_mae = make_daily_plot(
            args.csv, target, args.plot_output
        )
        print(f"Target day (KST): {target:%Y-%m-%d}")
        print(f"Training cutoff (KST): {target - timedelta(hours=1):%Y-%m-%d %H:%M}")
        print(f"Baseline MAE (10:00-23:00): {baseline_mae:.2f} bikes")
        print(f"Linear regression MAE (10:00-23:00): {linear_mae:.2f} bikes")
        print(f"Saved plot: {args.plot_output}")
        print("The model was fitted once using data through 09:00.")
        return

    (
        persistence,
        linear,
        training_count,
        ridge_alpha,
        persistence_mae,
        linear_mae,
    ) = make_predictions(args.csv, target)
    latest_known = target - timedelta(hours=1)
    latest_row = _read_known_rows(args.csv, latest_known)[-1]
    holiday = latest_row["Holiday"]
    if target.weekday() < 5:
        day_type = "weekday"
    else:
        day_type = "weekend"

    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "target_time_KST",
                "latest_known_time_KST",
                "training_rows",
                "ridge_alpha",
                "day_type",
                "holiday",
                "season",
                "functioning_day",
                "persistence_prediction",
                "linear_regression_prediction",
                "walk_forward_folds",
                "persistence_walk_forward_tuning_mae",
                "linear_regression_walk_forward_tuning_mae",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "target_time_KST": target.strftime("%Y-%m-%d %H:%M"),
                "latest_known_time_KST": latest_known.strftime("%Y-%m-%d %H:%M"),
                "training_rows": training_count,
                "ridge_alpha": ridge_alpha,
                "day_type": day_type,
                "holiday": holiday,
                "season": latest_row["Seasons"],
                "functioning_day": latest_row["Functioning Day"],
                "persistence_prediction": persistence,
                "linear_regression_prediction": linear,
                "walk_forward_folds": training_count - 3,
                "persistence_walk_forward_tuning_mae": round(persistence_mae, 2),
                "linear_regression_walk_forward_tuning_mae": round(linear_mae, 2),
            }
        )

    print(f"Target (KST): {target:%Y-%m-%d %H:%M}")
    print(f"Latest input (KST): {latest_known:%Y-%m-%d %H:%M}")
    print(f"Training rows: {training_count}")
    print(f"Persistence baseline: {persistence} bikes")
    print(f"Regularized linear regression: {linear} bikes (Ridge alpha={ridge_alpha})")
    print(f"Walk-forward tuning MAE (persistence): {persistence_mae:.2f} bikes")
    print(f"Walk-forward tuning MAE (linear regression): {linear_mae:.2f} bikes")
    print(f"Walk-forward folds: {training_count - 3}")
    print(f"Saved comparison: {args.output}")
    print("The target-hour row and later observations were not used.")


if __name__ == "__main__":
    main()
