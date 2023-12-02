# Population Forecasting with SARIMA
# author: E. Mas  

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
import pmdarima as pm
import multiprocessing as mp
import ntt #custom module
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from math import sqrt
from sklearn.metrics import mean_squared_error

# Parameters for area
MESH_ID = 503324732
AOI_NAME = "Kochi"
AOI_POLYGON = None
AOI_MESH = "./data/kochi_z01mesh.geojson"
EVENT_NAME = "KOCHI"
EVENT_DATE_START = "2023-11-01"
EVENT_DATE_MAIN = "2023-11-10"
EVENT_DATE_END = "2023-11-20"
FILE_PREFIX = "kochi"
SEASONALITY = 24
HOURS_TO_FORECAST = 3
PERCENTAGE_OF_DATA_FOR_TRAINING = 0.7

def auto_fit_model(data, seasonality=24, summary=False):
    # Seasonal - fit stepwise auto-ARIMA
    smodel = pm.auto_arima(
        data,
        start_p=1,
        start_q=1,
        test="adf",
        max_p=3,
        max_q=3,
        m=seasonality,
        start_P=0,
        seasonal=True,
        d=None,
        D=1,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )  # out_of_sample_size=6,scoring='mse'
    if summary:
        print(smodel.summary())
    return smodel


def fit_model(data, order=(0, 0, 0), seasonal_order=(0, 0, 0, 0)):
    # fit with ARIMA
    model = sm.tsa.arima.ARIMA(data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    return model, model_fit


def forecast_next_arima(model_fit, steps=1):
    return model_fit.forecast(steps)


def model_forecast(smodel, data, test, n_periods=3, name=0):
    fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = pd.date_range(data.index[-1], periods=n_periods, freq="H")

    # make series for plotting purpose
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)
    plot_forecast(
        data, test, n_periods, name, fitted_series, lower_series, upper_series
    )
    return


def plot_forecast(
    data, test, split_threshold, n_periods, x, fitted_series, figname, zoom=False
):
    basic_colors = ["r", "g", "b"]
    # Plot
    plt.figure(figsize=(15, 5))
    if x > 0:
        plt.plot(
            pd.concat([data, test.iloc[x - 1 : x]]),
            color="k",
            label="Train",
            zorder=1,
            marker="o",
            markersize=2,
        )
    plt.plot(test, color="gray", label="Test", ls="--", alpha=0.5, zorder=2)
    plt.plot(
        test.index[x : x + n_periods],
        fitted_series[x],
        color=basic_colors[0],
        label=f"Current forecast",
        alpha=0.8,
        zorder=3,
    )
    plt.plot(
        test.index[: len(fitted_series)],
        list(zip(*fitted_series))[0],
        color=basic_colors[0],
        label=f"Forecasted data (1 h.)",
        alpha=0.5,
        zorder=4,
        marker="o",
        markersize=2,
    )

    plt.plot(
        test.index[n_periods - 1 : len(fitted_series) + n_periods - 1],
        list(zip(*fitted_series))[n_periods - 1],
        color=basic_colors[n_periods - 1],
        label=f"Forecasted data ({n_periods} h.)",
        alpha=0.5,
        zorder=5,
        marker="o",
        markersize=2,
    )
    # assign categories
    categories = np.arange(0, n_periods)
    # use colormap
    # basic_colors = ["r", "g", "b"]
    if n_periods <= len(basic_colors):
        colormap = np.append(basic_colors[:n_periods], [])
    else:
        blocks = int(n_periods / len(basic_colors))
        residual = np.mod(n_periods, len(basic_colors))
        colormap = np.append(basic_colors * blocks, basic_colors[:residual])
    plt.scatter(
        test.index[x : x + n_periods],
        fitted_series[x],
        color=colormap[categories],
        label=f"Forecast No. {x}",
        zorder=6,
    )
    plt.ylabel("Population")
    plt.xlabel("Date and time")
    plt.legend()
    number = max([data.max().values, test.max().values]).item()
    ymax = round(number / 100) * 100
    plt.ylim(0, 1.2 * ymax)
    if zoom:
        plt.xlim(data.index[split_threshold - 3], test.index[-1])
    r2_0 = np.corrcoef(
        test.iloc[: len(fitted_series)].values.transpose()[0],
        list(zip(*fitted_series))[0],
    )[0][1]
    r2_1 = np.corrcoef(
        test.iloc[
            n_periods - 1 : len(fitted_series) + n_periods - 1
        ].values.transpose()[0],
        list(zip(*fitted_series))[n_periods - 1],
    )[0][1]
    plt.title(
        f"{n_periods}h forecast of {EVENT_NAME} event with SARIMA ($r^2_1$ = {np.round(r2_0,3)}, $r^2_{n_periods}$ = {np.round(r2_1,3)})"
    )
    plt.savefig(figname, dpi=300)
    plt.close()
    return


def save_outputs(case, predictions, errors, rmses):
    with open(f"{case.folderpath}/data/{FILE_PREFIX}_predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)

    with open(f"{case.folderpath}/data/{FILE_PREFIX}_errors.pkl", "wb") as f:
        pickle.dump(errors, f)

    with open(f"{case.folderpath}/data/{FILE_PREFIX}_rmses.pkl", "wb") as f:
        pickle.dump(rmses, f)
    return

def save_outputs_mesh(case, meshid, predictions, errors, rmses):
    with open(f"{case.folderpath}/data/{meshid}/{FILE_PREFIX}_predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)

    with open(f"{case.folderpath}/data/{meshid}/{FILE_PREFIX}_errors.pkl", "wb") as f:
        pickle.dump(errors, f)

    with open(f"{case.folderpath}/data/{meshid}/{FILE_PREFIX}_rmses.pkl", "wb") as f:
        pickle.dump(rmses, f)
    return

def forecast_area(case):
    # threshold
    th = int(PERCENTAGE_OF_DATA_FOR_TRAINING * case.pop.shape[0])
    # Create Training and Test
    data = case.pop[:th]
    obs = case.pop[th:]

    try:
        with open(
            f"{case.folderpath}/data/arima_{FILE_PREFIX}_split{10*PERCENTAGE_OF_DATA_FOR_TRAINING}_m{SEASONALITY}.pkl",
            "rb",
        ) as pkl:
            smodel = pickle.load(pkl)
        print("Pickle file loaded")
    except:
        t0 = time.time()
        print("Adjusting SARIMA model")
        smodel = auto_fit_model(data=data, seasonality=SEASONALITY, summary=False)
        print(f"Time: {time.time()-t0} s.")
        with open(
            f"{case.folderpath}/data/arima_{FILE_PREFIX}_split{10*PERCENTAGE_OF_DATA_FOR_TRAINING}_m{SEASONALITY}.pkl",
            "wb",
        ) as pkl:
            pickle.dump(smodel, pkl)
    model_order = smodel.order
    seasonal_order = smodel.seasonal_order
    print(model_order, seasonal_order)

    predictions = list()
    errors = list()
    rmses = list()
    
    for t in range(obs.shape[0] - HOURS_TO_FORECAST):
        print("=" * 10)
        model = sm.tsa.arima.ARIMA(
            data, order=model_order, seasonal_order=seasonal_order
        )

        start = time.time()
        model_fit = model.fit(low_memory=True)
        print(f"Model Fit: {time.time()-start} s")

        start = time.time()
        forecasted = forecast_next_arima(model_fit, steps=HOURS_TO_FORECAST)
        print(f"Forecast: {time.time()-start} s")

        predictions.append(forecasted.values)
        errors.append(
            forecasted.values - obs.iloc[t : t + HOURS_TO_FORECAST].values.transpose()
        )
        rmses.append(
            sqrt(
                mean_squared_error(
                    case.pop[th : th + len(predictions)],
                    list(list(zip(*predictions))[0]),
                )
            )
        )

        start = time.time()
        figname = Path(case.folderpath, f"figures/{t:04d}.png")
        plot_forecast(
            data=data,
            test=obs,
            split_threshold=th,
            n_periods=HOURS_TO_FORECAST,
            x=t,
            fitted_series=predictions,
            figname=figname,
            zoom=True,
        )
        print(f"Plot: {time.time()-start} s")
        print(
            f"{t}:{obs.shape[0]} - forecasted: {forecasted.values} ; real: {obs.iloc[t:t+HOURS_TO_FORECAST].values[0]} ; error: {errors[-1]} ; rmse: {rmses[-1]}"
        )
        data = pd.concat([data, obs.iloc[t : t + 1]])
        smodel.update(obs.iloc[t].values)
        model_order = smodel.order
        seasonal_order = smodel.seasonal_order
    save_outputs(case=case, predictions=predictions, errors=errors, rmses=rmses)
    case.make_video()

    # plot errors
    N = case.pop.mean()
    err = [errors[i][0][0] / N for i in range(len(errors))]
    plt.figure(figsize=(20, 6))
    plt.plot(np.arange(0, len(errors)), err)
    plt.plot(np.arange(0, len(errors)), [np.mean(err)] * len(errors))
    plt.plot(np.arange(0, len(errors)), [np.mean(err) + np.std(err)] * len(errors))
    plt.plot(np.arange(0, len(errors)), [np.mean(err) - np.std(err)] * len(errors))
    plt.ylim(-1, 1)
    figname = Path(case.folderpath, f"plots/errors.png")
    plt.savefig(figname, dpi=300)


def forecast_mesh(case,meshid,pop):
    # threshold
    th = int(PERCENTAGE_OF_DATA_FOR_TRAINING * pop.shape[0])
    # Create Training and Test
    data = pd.DataFrame(pop[:th])
    obs = pd.DataFrame(pop[th:])
    

    try:
        with open(
            f"{case.folderpath}/data/{meshid}/arima_{FILE_PREFIX}_split{10*PERCENTAGE_OF_DATA_FOR_TRAINING}_m{SEASONALITY}.pkl",
            "rb",
        ) as pkl:
            smodel = pickle.load(pkl)
        print("Pickle file loaded")
    except:
        t0 = time.time()
        print("Adjusting SARIMA model")
        smodel = auto_fit_model(data=data, seasonality=SEASONALITY, summary=False)
        print(f"Time: {time.time()-t0} s.")
        with open(
            f"{case.folderpath}/data/{meshid}/arima_{FILE_PREFIX}_split{10*PERCENTAGE_OF_DATA_FOR_TRAINING}_m{SEASONALITY}.pkl",
            "wb",
        ) as pkl:
            pickle.dump(smodel, pkl)
    model_order = smodel.order
    seasonal_order = smodel.seasonal_order
    print(model_order, seasonal_order)

    predictions = list()
    errors = list()
    rmses = list()
    
    for t in range(obs.shape[0] - HOURS_TO_FORECAST):
        print("=" * 10)
        model = sm.tsa.arima.ARIMA(
            data, order=model_order, seasonal_order=seasonal_order,
            enforce_stationarity=False
        )

        start = time.time()
        model_fit = model.fit(low_memory=True)
        print(f"Model Fit: {time.time()-start} s")

        start = time.time()
        forecasted = forecast_next_arima(model_fit, steps=HOURS_TO_FORECAST)
        print(f"Forecast: {time.time()-start} s")

        predictions.append(forecasted.values)
        errors.append(
            forecasted.values - obs.iloc[t : t + HOURS_TO_FORECAST].values.transpose()
        )
        rmses.append(
            sqrt(
                mean_squared_error(
                    pop[th : th + len(predictions)],
                    list(list(zip(*predictions))[0]),
                )
            )
        )

        start = time.time()
        figname = Path(case.folderpath, f"figures/{meshid}/{t:04d}.png")
        plot_forecast(
            data=data,
            test=obs,
            split_threshold=th,
            n_periods=HOURS_TO_FORECAST,
            x=t,
            fitted_series=predictions,
            figname=figname,
            zoom=True,
        )
        print(f"Plot: {time.time()-start} s")
        print(
            f"{t}:{obs.shape[0]} - forecasted: {forecasted.values} ; real: {obs.iloc[t:t+HOURS_TO_FORECAST].values[0]} ; error: {errors[-1]} ; rmse: {rmses[-1]}"
        )
        data = pd.concat([data, obs.iloc[t : t + 1]])
        smodel.update(obs.iloc[t].values)
        model_order = smodel.order
        seasonal_order = smodel.seasonal_order
    save_outputs_mesh(case=case, meshid=meshid, predictions=predictions, errors=errors, rmses=rmses)
    case.make_video_mesh(meshid)

    # plot errors
    N = case.pop.mean()
    err = [errors[i][0][0] / N for i in range(len(errors))]
    plt.figure(figsize=(20, 6))
    plt.plot(np.arange(0, len(errors)), err)
    plt.plot(np.arange(0, len(errors)), [np.mean(err)] * len(errors))
    plt.plot(np.arange(0, len(errors)), [np.mean(err) + np.std(err)] * len(errors))
    plt.plot(np.arange(0, len(errors)), [np.mean(err) - np.std(err)] * len(errors))
    plt.ylim(-1, 1)
    figname = Path(case.folderpath, f"plots/{meshid}/errors.png")
    plt.savefig(figname, dpi=300)
    
def format_output(case):
    all_data = ntt.read_object(f'{case.folderpath}/data/{FILE_PREFIX}_nttclass_ftype0.pickle')
    os.makedirs(Path(ROOT, "output"), exist_ok=True)
    split = PERCENTAGE_OF_DATA_FOR_TRAINING
    dict_pred = {}
    th = int(split * len(all_data.popm))
    train = all_data.popm.iloc[:th]
    obs = all_data.popm.iloc[th:]
    dt_start = train.index[-1].strftime("%Y%m%d%H%M")
    dt_end = obs.index[-HOURS_TO_FORECAST-2].strftime("%Y%m%d%H%M")
    date_index = pd.date_range(dt_start, dt_end, freq='H').strftime("%Y%m%d%H%M")

    for t, dt in enumerate(date_index):
        for meshid in all_data.list_mesh:
            pred = ntt.read_object(f'{case.folderpath}/data/{meshid}/{FILE_PREFIX}_predictions.pkl')
            df_pred = pd.DataFrame(pred, columns=['1h','2h','3h'])
            df_true = pd.DataFrame(all_data.popm[meshid])
            dict_pred[meshid] = df_pred.iloc[t]
        temp = pd.DataFrame(dict_pred).T.astype(int)
        temp['meshid'] = temp.index
        temp = temp.reset_index(drop=True)
        temp = temp[['meshid','1h','2h','3h']]
        temp.to_csv(f'{case.folderpath}/output/{dt}.csv', index=False)
    

if __name__ == "__main__":
    print("Starting")
    process_count = mp.cpu_count()
    print(f"Number of processes: {process_count}")
    # Load data
    case = ntt.MobileData(
        one_mesh=MESH_ID,
        aoi_name=AOI_NAME,
        aoi_pol=AOI_POLYGON,
        aoi_mesh=AOI_MESH,
        event_name=EVENT_NAME,
        dt_start=EVENT_DATE_START,
        dt_main=EVENT_DATE_MAIN,
        dt_end=EVENT_DATE_END,
        fpfx=FILE_PREFIX
    )
    print("Data loaded")
    case.plot_population(save=True)
    case.get_population_by_mesh()
    for meshid in case.list_mesh:
        print(f"Forecasting mesh: {meshid}")
        # PARAMETERS
        ROOT = case.folderpath
        # FIGURES_FOLDER = Path(ROOT, "figures")
        # OUTPUTS_FOLDER = Path(ROOT, "data")
        os.makedirs(Path(ROOT, "data", str(meshid)), exist_ok=True)
        os.makedirs(Path(ROOT, "figures", str(meshid)), exist_ok=True)
        os.makedirs(Path(ROOT, "plots", str(meshid)), exist_ok=True)
        forecast_mesh(case,meshid,case.popm[meshid])
    format_output(case)
    # forecast_area(case)
