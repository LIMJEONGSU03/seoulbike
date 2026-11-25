# Seoul Bike rental forecast

## Goal and data

- Forecast the rental count for `2017-12-01 10:00 KST`.
- Source: `SeoulBikeData.csv`, with dates parsed as `DD/MM/YYYY`.
- The source covers `2017-12-01` through `2018-11-30`. It contains no records before December 1, 2017.
- For the 10:00 forecast, only observations through 09:00 are inputs. The 10:00 row and later observations are not used to fit or predict the 10:00 value.

## Forecast methods

### Persistence baseline

Use the latest known rental count as the next-hour forecast. At 10:00, this is the 09:00 count: **490 bikes**.

### Regularized linear regression

The script fits a standardized Ridge regression. For each training example, features are drawn from the hour before its target:

- Rental counts at the feature hour and one and two hours earlier.
- Weather observed at the feature hour: temperature, humidity, wind speed, visibility, dew point temperature, solar radiation, rainfall, and snowfall.
- Known calendar/context features: target hour, weekday, holiday status, functioning-day status, and winter indicator.

Only seven next-hour training examples can be formed from data available through 09:00 (targets 03:00–09:00). The model is trained once; it is not retrained during evaluation. Ridge strength is selected from a small candidate set using chronological walk-forward validation on these pre-10:00 examples.

The resulting 10:00 linear-regression forecast is **540 bikes**. This estimate is uncertain: seven training examples are very limited, and the same small set is used to select Ridge strength.

## Results

### 10:00 forecast

| Method | Forecast |
| --- | ---: |
| Persistence baseline | 490 |
| Regularized linear regression (Ridge) | 540 |

The initial informal estimate of about 400 bikes was a rough judgment, not a trained model prediction.

### December 1 plot and MAE

The chart shows actual rentals from 09:00 to 23:00 and forecasts from 10:00 to 23:00. Forecasts are one-step-ahead: each later forecast can use observations that would have become available by that time. The fitted regression model remains fixed after the 09:00 cutoff.

MAE is calculated over the **14 forecast hours, 10:00–23:00**:

| Method | MAE |
| --- | ---: |
| Persistence baseline | 87.21 bikes |
| Regularized linear regression (Ridge) | 370.57 bikes |

The persistence baseline performed better on this day. These results are specific to this date and should not be treated as a general performance guarantee.

The earlier four-fold walk-forward tuning MAEs were 317.50 bikes for persistence and 273.21 bikes for Ridge. Ridge strength was selected on those same folds, so those values are tuning diagnostics, not independent test scores.

## Files and commands

- `baseline_model.py`: model, forecast, and plotting code.
- `SeoulBike.csv`: saved comparison for the 10:00 forecast.
- `SeoulBikeData.png`: plot with actual values in black, persistence in gray dashed, and Ridge predictions in blue.
- `SeoulBikeData.csv`: source dataset.

Run the single-hour comparison:

```powershell
python baseline_model.py
```

Generate the December 1 plot and its 10:00–23:00 MAEs:

```powershell
python baseline_model.py --plot-day
```

The script requires Python, scikit-learn, and matplotlib. It reads the CSV as Windows-1252 and treats the recorded times as KST.
