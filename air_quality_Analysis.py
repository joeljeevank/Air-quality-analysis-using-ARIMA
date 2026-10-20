import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path



###finall........

# LOAD DATA AND BASIC CLEANING
file_path = r"D:\MCA\SEM3\Assignment\DAVFeliciaMam\Air_quality.csv"
output_dir = Path("air_quality_complete_output")
output_dir.mkdir(exist_ok=True)

df = pd.read_csv(file_path)

print("Original Shape:", df.shape)
print("\nStatistical Summary:")
print(df.describe(include="all"))

df = df.replace(["NA", "N/A", "na", "n/a", "", " "], np.nan)
df = df.replace(r"^\s*$", np.nan, regex=True)

pollutants = ["PM2.5", "PM10", "NO2", "NH3", "SO2", "CO", "O3"]
available_pollutants = [column for column in pollutants if column in df.columns]

print("\n--- Missing Values Before Handling ---")
print(df.isnull().sum())

if "Location" not in df.columns or "Timestamp" not in df.columns:
    raise ValueError("The CSV must contain 'Location' and 'Timestamp' columns.")

df["Location"] = df["Location"].astype(str).str.strip()
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"], format="%d-%m-%Y", errors="coerce"
)

for column in available_pollutants:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df.loc[df["Location"].isin(["", "nan", "None"]), "Location"] = np.nan
df = df.dropna(subset=["Timestamp", "Location"]).copy()
df = df.sort_values(["Location", "Timestamp"])

# Interpolate missing pollutant values separately for each location
for column in available_pollutants:
    df[column] = df.groupby("Location")[column].transform(
        lambda values: values.interpolate(method="linear", limit_direction="both")
    )

print("\n--- Missing Values After Handling ---")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nLocations:")
print(df["Location"].unique())


# OUTLIER DETECTION WITH IQR AND BOXPLOTS


boxplot_data = []
boxplot_columns = []

for column in available_pollutants:
    values = df[column].dropna()
    if values.empty:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[column] < lower) | (df[column] > upper)]

    print(column, "Outliers:", len(outliers))
    boxplot_columns.append(column)
    boxplot_data.append(values)

if boxplot_columns:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, column in enumerate(boxplot_columns):
        axes[i].boxplot(boxplot_data[i])
        axes[i].set_title("Box Plot - " + column)
        axes[i].set_ylabel("Concentration")
        axes[i].set_xticks([])
    for i in range(len(boxplot_columns), len(axes)):
        axes[i].set_visible(False)
    plt.suptitle("Outlier Detection for Air Pollutants", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "pollutant_outlier_boxplots.png", dpi=150)
    plt.show()
    plt.close()


# DESCRIPTIVE STATISTICS


df["Year"] = df["Timestamp"].dt.year
df["Month"] = df["Timestamp"].dt.month
df["Day"] = df["Timestamp"].dt.day

if available_pollutants:
    df["Average_Pollution"] = df[available_pollutants].mean(axis=1)

print("\n--- AFTER FEATURE ENGINEERING ---")
print(df[["Timestamp", "Year", "Month", "Day"]].head())

if "Average_Pollution" in df.columns:
    print("\n--- AVERAGE POLLUTION FEATURE ---")
    print(df[["Timestamp", "Location", "Average_Pollution"]].head())

print("\nPollutant descriptive statistics:")
print(df[available_pollutants].describe())

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)
pd.set_option("display.expand_frame_repr", False)


# POLLUTION BY LOCATION


print("\n========== A. POLLUTION BY LOCATION ==========")
location_pollution = df.groupby("Location")[available_pollutants].mean()
print("\nAverage Pollution by Location:")
print(location_pollution)
print("\nNumber of locations:", df["Location"].nunique())
print(df["Location"].unique())

if not location_pollution.empty:
    location_pollution.plot(kind="bar", figsize=(14, 7))
    plt.title("Average Pollution by Location")
    plt.xlabel("Location")
    plt.ylabel("Average Concentration")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Pollutants")
    plt.tight_layout()
    plt.savefig(output_dir / "average_pollution_by_location.png", dpi=150)
    plt.show()
    plt.close()


# POLLUTION BY YEAR AND INDIVIDUAL POLLUTANT TRENDS


print("\n========== B. POLLUTION BY YEAR ==========")
yearly_pollution = df.groupby("Year")[available_pollutants].mean()
print("\nAverage Pollution by Year:")
print(yearly_pollution)
print("\nYears available in dataset:")
print(yearly_pollution.index.tolist())

if not yearly_pollution.empty:
    yearly_pollution.plot(kind="line", marker="o", figsize=(12, 6))
    plt.title("Average Pollution Trend by Year")
    plt.xlabel("Year")
    plt.ylabel("Average Concentration")
    plt.xticks(yearly_pollution.index)
    plt.legend(title="Pollutants")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "average_pollution_by_year.png", dpi=150)
    plt.show()
    plt.close()

print("\n========== C. POLLUTANT TRENDS ==========")
for pollutant in available_pollutants:
    values = yearly_pollution[pollutant].dropna()
    if len(values) < 2:
        print(pollutant, "-> Not enough yearly data to determine a trend.")
        continue

    first_value = values.iloc[0]
    last_value = values.iloc[-1]
    change = last_value - first_value

    if change > 0:
        trend = "Increased"
    elif change < 0:
        trend = "Decreased"
    else:
        trend = "No Change"

    yearly_values = ", ".join(
        f"{year}: {value:.2f}" for year, value in values.items()
    )
    print(f"{pollutant}: {yearly_values} | Change: {change:.2f} | Trend: {trend}")

    plt.figure(figsize=(9, 5))
    plt.plot(values.index, values.values, marker="o")
    plt.title(pollutant + " Trend by Year")
    plt.xlabel("Year")
    plt.ylabel("Average " + pollutant)
    plt.xticks(values.index)
    plt.grid(True)
    plt.tight_layout()
    safe_pollutant = "".join(ch if ch.isalnum() else "_" for ch in pollutant)
    plt.savefig(output_dir / f"trend_{safe_pollutant}.png", dpi=150)
    plt.show()
    plt.close()


# CORRELATION ANALYSIS


print("\n========== D. CORRELATION ANALYSIS ==========")
correlation = df[available_pollutants].corr()
print("\nCorrelation Matrix:")
print(correlation)

if not correlation.empty:
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )
    plt.title("Correlation Between Air Pollutants")
    plt.tight_layout()
    plt.savefig(output_dir / "pollutant_correlation_heatmap.png", dpi=150)
    plt.show()
    plt.close()


# FIVE ANALYTICAL QUESTIONS



print("5 ANALYTICAL QUESTIONS")


print("\nQUESTION 1: Which location has the highest average concentration of each pollutant?")
for pollutant in available_pollutants:
    values = location_pollution[pollutant].dropna()
    if not values.empty:
        print(pollutant, "->", values.idxmax(), "| Average =", round(values.max(), 2))

print("\nQUESTION 2: How do average pollutant levels change from 2020 to 2024?")
if 2020 in yearly_pollution.index and 2024 in yearly_pollution.index:
    for pollutant in available_pollutants:
        value_2020 = yearly_pollution.loc[2020, pollutant]
        value_2024 = yearly_pollution.loc[2024, pollutant]
        print(
            pollutant, "-> 2020:", round(value_2020, 2),
            "| 2024:", round(value_2024, 2),
            "| Change:", round(value_2024 - value_2020, 2)
        )
else:
    print("Cannot compare 2020 and 2024 because one or both years are missing.")

print("\nQUESTION 3: How does PM2.5 change from 2020 to 2024?")
if "PM2.5" in available_pollutants and 2020 in yearly_pollution.index and 2024 in yearly_pollution.index:
    pm25_change = yearly_pollution.loc[2024, "PM2.5"] - yearly_pollution.loc[2020, "PM2.5"]
    print("PM2.5 in 2020:", round(yearly_pollution.loc[2020, "PM2.5"], 2))
    print("PM2.5 in 2024:", round(yearly_pollution.loc[2024, "PM2.5"], 2))
    print("Change in PM2.5:", round(pm25_change, 2))
    if pm25_change > 0:
        print("Answer: PM2.5 increased from 2020 to 2024.")
    elif pm25_change < 0:
        print("Answer: PM2.5 decreased from 2020 to 2024.")
    else:
        print("Answer: PM2.5 remained unchanged.")
else:
    print("Cannot answer because PM2.5 data for both 2020 and 2024 is not available.")

print("\nQUESTION 4: Which pollutants show increasing or decreasing trends between 2020 and 2024?")
if 2020 in yearly_pollution.index and 2024 in yearly_pollution.index:
    for pollutant in available_pollutants:
        value_2020 = yearly_pollution.loc[2020, pollutant]
        value_2024 = yearly_pollution.loc[2024, pollutant]
        if value_2024 > value_2020:
            trend = "Increasing"
        elif value_2024 < value_2020:
            trend = "Decreasing"
        else:
            trend = "No Change"
        print(pollutant, "->", trend)
else:
    print("Cannot compare trends because one or both years are missing.")

print("\nQUESTION 5: What relationships exist between the different pollutants?")
print("\nCorrelation Matrix:")
print(correlation)
if len(available_pollutants) > 1:
    corr_pairs = correlation.unstack()
    corr_pairs = corr_pairs[
        corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)
    ]
    corr_pairs = corr_pairs.sort_values(ascending=False)
    print("\nStrongest Positive Relationships (duplicate pairs may appear):")
    print(corr_pairs.head(10))




# ARIMA MODEL: EVALUATION AND FUTURE PREDICTION


print("\n=== ARIMA PM2.5 EVALUATION AND FORECASTING ===")

locations = [
    "Chennai - Alandur Bus Depot",
    "Bengaluru - Silk Board",
    "Delhi - Punjabi Bagh",
    "Hyderabad - Central University",
    "Kolkata - Rabindra Bharati University",
    "Lucknow - Lalbagh",
    "Mumbai - Chhatrapati Shivaji International Airport (T2)",
    "Visakhapatnam - GVM Corporation"
]

results = []
future_results = []

if "PM2.5" not in df.columns:
    print("ARIMA was skipped because PM2.5 is not available.")

else:
    for location in locations:

        
        print("Location:", location)
        

        
        # SELECT LOCATION DATA
     

        location_data = df[
            df["Location"] == location
        ].copy()

        location_data = location_data[
            ["Timestamp", "PM2.5"]
        ].dropna()

        location_data = location_data.sort_values(
            "Timestamp"
        )

        # Handle duplicate dates
        location_data = location_data.groupby(
            "Timestamp", as_index=False
        )["PM2.5"].mean()

        if location_data.empty:
            print("No valid data for:", location)
            continue

        # Set Timestamp as index and use daily frequency
        location_data.set_index(
            "Timestamp", inplace=True
        )

        location_data = location_data.asfreq("D")

        # Fill missing daily PM2.5 values
        location_data["PM2.5"] = (
            location_data["PM2.5"]
            .interpolate(
                method="linear",
                limit_direction="both"
            )
        )

        series = location_data["PM2.5"].dropna()

        # Check whether sufficient data is available
        if len(series) < 100:
            print(
                "Not enough data for:",
                location,
                "- Records:",
                len(series)
            )
            continue

        safe_location = "".join(
            char if char.isalnum() else "_"
            for char in location
        ).strip("_")

        try:

            
            # TRAIN / TEST SPLIT (80/20)
           

            train_size = int(len(series) * 0.80)

            train = series.iloc[:train_size]
            test = series.iloc[train_size:]

           
            # BUILD ARIMA MODEL
           

            model = ARIMA(
                train,
                order=(1, 1, 1)
            )

            model_fit = model.fit()

           
            # FORECAST TEST DATA
            

            forecast = model_fit.forecast(
                steps=len(test)
            )

            forecast.index = test.index

           
            # EVALUATION: MAE, RMSE AND MAPE
           

            mae = mean_absolute_error(
                test, forecast
            )

            rmse = np.sqrt(
                mean_squared_error(
                    test, forecast
                )
            )

            # Avoid division by zero in MAPE
            non_zero = test != 0

            if non_zero.any():
                mape = np.mean(
                    np.abs(
                        (
                            test[non_zero]
                            - forecast[non_zero]
                        ) / test[non_zero]
                    )
                ) * 100
            else:
                mape = np.nan

            print("Training data :", len(train))
            print("Testing data  :", len(test))
            print("MAE           :", round(mae, 3))
            print("RMSE          :", round(rmse, 3))

            if pd.notna(mape):
                print("MAPE          :", round(mape, 3), "%")
            else:
                print("MAPE          : Not available")

           
            # SAVE EVALUATION RESULTS
        

            results.append({
                "Location": location,
                "Training data": len(train),
                "Testing data": len(test),
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape
            })

            # Save actual and predicted test values
            test_results = pd.DataFrame({
                "Location": location,
                "Timestamp": test.index,
                "Actual_PM2.5": test.to_numpy(),
                "Predicted_PM2.5": forecast.to_numpy()
            })

            test_results.to_csv(
                output_dir /
                f"test_predictions_{safe_location}.csv",
                index=False
            )

            
            # PLOT TRAINING, ACTUAL AND PREDICTED DATA
        

            plt.figure(figsize=(12, 6))

            plt.plot(
                train.index,
                train,
                label="Training Data"
            )

            plt.plot(
                test.index,
                test,
                label="Actual PM2.5"
            )

            plt.plot(
                forecast.index,
                forecast,
                label="ARIMA Prediction"
            )

            plt.title(
                "ARIMA PM2.5 Forecast - " + location
            )

            plt.xlabel("Date")
            plt.ylabel("PM2.5")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plt.savefig(
                output_dir /
                f"arima_test_forecast_{safe_location}.png",
                dpi=150
            )

            plt.show()
            plt.close()

            
            # 8. FORECAST THE NEXT 30 DAYS
            

            # Train a new model using all historical data
            full_model = ARIMA(
                series,
                order=(1, 1, 1)
            )

            full_model_fit = full_model.fit()

            future_prediction = full_model_fit.forecast(
                steps=30
            )

            # Generate the next 30 dates
            future_dates = pd.date_range(
                start=series.index.max()
                + pd.Timedelta(days=1),
                periods=30,
                freq="D"
            )

            # Create future prediction table
            future_df = pd.DataFrame({
                "Location": location,
                "Date": future_dates,
                "Predicted_PM2.5":
                    future_prediction.to_numpy()
            })

            future_results.append(future_df)

            print("\nNext 30 Days PM2.5 Prediction:")
            print(future_df.to_string(index=False))

            # Save future predictions for this location
            future_df.to_csv(
                output_dir /
                f"future_30_day_prediction_{safe_location}.csv",
                index=False
            )

            
            # PLOT THE 30-DAY FUTURE FORECAST
           

            recent_history = series.tail(90)

            plt.figure(figsize=(12, 6))

            plt.plot(
                recent_history.index,
                recent_history.values,
                label="Historical PM2.5"
            )

            plt.plot(
                future_dates,
                future_prediction.to_numpy(),
                label="Next 30 Days Prediction",
                linestyle="--",
                marker="o",
                markersize=3
            )

            plt.axvline(
                series.index.max(),
                linestyle=":",
                label="Forecast Starts"
            )

            plt.title(
                "30-Day Future PM2.5 Prediction - "
                + location
            )

            plt.xlabel("Date")
            plt.ylabel("PM2.5")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plt.savefig(
                output_dir /
                f"future_30_day_forecast_{safe_location}.png",
                dpi=150
            )

            plt.show()
            plt.close()

        except Exception as error:
            print(
                "Could not complete ARIMA analysis for",
                location
            )
            print("Error:", error)



# FINAL ARIMA EVALUATION RESULTS TABLE


print("\n== ARIMA RESULTS - ALL LOCATIONS ===")

results_df = pd.DataFrame(results)

if results_df.empty:
    print("No evaluation results were generated.")

else:
    print(results_df.to_string(index=False))

    results_df.to_csv(
        output_dir / "arima_evaluation_results.csv",
        index=False
    )



# COMBINE ALL 30-DAY FUTURE PREDICTIONS


print("\n=== 30-DAY PREDICTIONS - ALL LOCATIONS ===")

if future_results:

    all_future_predictions = pd.concat(
        future_results,
        ignore_index=True
    )

    print(
        all_future_predictions.to_string(index=False)
    )

    all_future_predictions.to_csv(
        output_dir /
        "all_locations_30_day_future_predictions.csv",
        index=False
    )

else:
    print("No future predictions were generated.")


