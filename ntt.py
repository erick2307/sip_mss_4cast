import glob
import os
import time
import pickle
import cloudpickle
import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib
import zipfile
from dotenv import load_dotenv
from datetime import datetime, date, timedelta
from cmcrameri import cm
from pathlib import Path
from tqdm import tqdm

matplotlib.use("Agg")  # to solve issue on out of memory
plt.style.use("default")

def configure():
    load_dotenv()

def read_object(path):
    """To load data from a pickle"""
    with open(Path(path), "rb") as handle:
        obj = pickle.load(handle)
    return obj

# search for the ZIP files on a specific year.month.date
def zippedfiles(year=2022, month=10, date=1):
    folder = Path(os.getenv('HOME_DIR'),f"{year}{month:02d}{date:02d}")
    zipfiles = sorted(glob.glob(os.path.join(folder, "*.zip")))
    return zipfiles


# extract all zippedfiles into a new folder
def extractfiles(year=2019, month=1, date=1):
    datefolder = f"{year}{month:02d}{date:02d}"
    if not os.path.exists(
        os.path.join(os.getenv('HOME_DIR'), datefolder)
    ):
        return f"{os.getenv('HOME_DIR')}{datefolder} does not exist"
    outfolder = Path(os.getenv('HOME_DIR'),f"{year}_csv/{datefolder}")
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    zipfiles = zippedfiles(year, month, date)
    for f in tqdm(zipfiles, desc=f"{datefolder}", position=1):
        with zipfile.ZipFile(f, "r") as zip_ref:
            zip_ref.extractall(f"{outfolder}")
    return


def unzip(start, end):
    dates = pd.date_range(start, end)
    print(f"Extracting {len(dates)} days ...")
    file_log = tqdm(total=0, position=2, bar_format="{desc}")
    for date in tqdm(dates, desc="Dates", position=0):
        s = time.time()
        extractfiles(date.year, date.month, date.day)
        t1 = time.time() - s
        file_log.set_description_str(f"Folder:{date}, Loadtime:{t1}s")
    return


class MobileData:
    def __init__(
        self,
        one_mesh,
        aoi_name,
        aoi_pol,
        aoi_mesh,
        event_name,
        dt_start,
        dt_main,
        dt_end,
        fpfx,
    ):
        configure()
        self.dfd = {}  # Dataframe of one day data (dynamic)
        self.gdfp = {}  # GeoDataframe of period data (fix)
        self.dft = {}  # Dataframe temporal (when reading from pickle)
        self.bbox = set()
        self.pop = pd.DataFrame()
        self.fpfx = fpfx
        self.one_mesh = one_mesh
        self.aoi_name = aoi_name
        self.folderpath = Path(
            self.aoi_name, str(datetime.now().strftime("%Y%m%d%H%M%S"))
        )
        self._create_directory(self.folderpath)
        self._create_directory(Path(self.folderpath, "plots"))
        self._create_directory(Path(self.folderpath, "figures"))
        self._create_directory(Path(self.folderpath, "data"))
        if aoi_pol == None and aoi_mesh == None:
            raise OSError("No file found. Provide AOI_POLYGON or AOI_MESH")
        if aoi_pol != None:
            self.aoi_pol = gpd.read_file(aoi_pol)
        else:
            print("No AOI_POLYGON path provided. Trying with AOI_MESH...")
        if aoi_mesh == None:
            print("Creating mesh file...")
            self.aoi_mesh = gpd.read_file(os.getenv('JAPAN_MESH4'),mask=self.aoi_pol)
        else:
            print("Reading mesh from file...")
            self.aoi_mesh = gpd.read_file(aoi_mesh)
        self.aoi_mesh.to_file(
            Path(
                self.folderpath, "data", f"./{self.fpfx}_{self.aoi_name}_mesh.geojson"
            ),
            driver="GeoJSON",
        )
        print("AOI_MESH file saved in 'data' folder.")
        self.list_mesh = list(self.aoi_mesh["MESH4_ID"].values.astype("int64"))
        self.e_name = event_name
        self.sdate = pd.to_datetime(dt_start)
        if dt_main == None:
            self.mdate = pd.to_datetime(dt_start)
            print("Main date set to same as Start date.")
        else:
            self.mdate = pd.to_datetime(dt_main)
        self.edate = pd.to_datetime(dt_end)
        self.date_idx = pd.date_range(self.sdate, self.edate, freq="d")
        self.hour_idx = pd.date_range(self.sdate, self.edate, freq="H")
        print(
            f"""
              ===== {self.e_name} =====
              1. Event Name     ==> {self.e_name}
              2. Folder         ==> {self.aoi_name}
              3. Working CRS    ==> {self.aoi_mesh.crs.to_string()} 
              4. One mesh       ==> {self.one_mesh}
              5. Start Date     ==> {self.sdate}
              6. Main Date      ==> {self.mdate}
              7. End Date       ==> {self.edate}
              ========================
              """
        )
        return

    def _create_directory(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"Folder {path} created")
        return

    def read_one_day_data(self, y=2019, m=1, d=1, ftype=0):
        """Read one day data and
        return a dictionary
        of the day key: hour"""
        outfolder = Path(os.getenv('HOME_DIR'),f"{y}_csv/{y}{m:02d}{d:02d}")
        # create csv path and filenames list
        csvfiles = sorted(glob.glob(os.path.join(outfolder, f"*{ftype}.csv")))
        # read all data
        self.dfd = {k: pd.read_csv(c) for k, c in enumerate(csvfiles)}
        return

    def read_period_data(self, ftype=0):
        for date in self.date_idx:
            y = date.date().year
            m = date.date().month
            d = date.date().day
            self.read_one_day_data(y, m, d, ftype)
            for hour in self.dfd.keys():
                gdf = self.intersect_data_mesh(self.dfd[hour], self.aoi_mesh)
                self.gdfp[f"{y}{m:02d}{d:02d}{hour:02d}00"] = gdf
        # might be not so efficient doing it here
        self._max_min_pop_period()
        self._store_dict(self.gdfp, name=f"gdf_dict_ftype{ftype}")
        return

    def get_population(self, ftype=0):
        if self.gdfp == {}:
            print("Reading period data...")
            self.read_period_data(ftype)
        print("Calculating population in period...")
        pop_dict = {}
        for i, key in enumerate(self.gdfp.keys()):
            df = self.gdfp[key]
            pop_dict[key] = df["population"].sum()
            self.pop = pd.DataFrame.from_dict(
                pop_dict, orient="index", columns=["population"]
            )
            # Converting the index as date
            self.pop.index = pd.to_datetime(self.pop.index)
        self._store_object(name=f"nttclass_ftype{ftype}")
        return
    
    def get_population_by_mesh(self, ftype=0):
        if self.gdfp == {}:
            print("Reading period data...")
            self.read_period_data(ftype)
        print("Calculating population by mesh in period...")
        popm_dict = {}
        for mesh in self.list_mesh:
            popm_dict[mesh] = {}
            for i, key in enumerate(self.gdfp.keys()):
                df = self.gdfp[key]
                popm_dict[mesh][key] = df[df["MESH4_ID"] == mesh]["population"].sum()
        self.popm = pd.DataFrame.from_dict(popm_dict)
        self.popm.index = pd.to_datetime(self.popm.index)
        self._store_object(name=f"nttclass_ftype{ftype}")
        return

    def _store_dict(self, dfd, name="data"):
        """To store data in a pickle"""
        with open(
            Path(self.folderpath, "data", f"{self.fpfx}_{name}.pickle"), "wb"
        ) as handle:
            pickle.dump(dfd, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def _store_object(self, name="nttclass_ftype0"):
        """To class object in a pickle"""
        with open(
            Path(self.folderpath, "data", f"{self.fpfx}_{name}.pickle"), "wb"
        ) as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def _store_object_cloudpickle(self, name="nttclass_ftype0"):
        """To class object in a pickle"""
        with open(
            Path(self.folderpath, "data", f"{self.fpfx}_{name}.pickle"), "wb"
        ) as handle:
            cloudpickle.dump(self, handle)
        return

    def read_dict(self, path, name="data"):
        """To load data from a pickle"""
        with open(Path(path, f"{self.fpfx}_{name}.pickle"), "rb") as handle:
            self.dft = pickle.load(handle)
        return self.dft

    def create_date_idx(self, start="20190101", end="20200101", freq="d"):
        """To create a set date index"""
        sdate = date(int(start[:4]), int(start[4:6]), int(start[-2:]))  # start date
        edate = date(
            int(end[:4]), int(end[4:6]), int(end[-2:])
        )  # end date (not inclusive in this pandas version)
        if freq == "d":
            date_idx = (
                pd.date_range(sdate, edate - timedelta(days=1), freq=freq)
                .strftime("%Y%m%d")
                .to_list()
            )
        elif freq == "H":
            date_idx = (
                pd.date_range(sdate, edate, freq=freq)
                .strftime("%Y%m%d%H")
                .to_list()[:-1]
            )
        else:
            date_idx = None
            print('input "freq"')
        return date_idx

    def read_mesh(self, path):
        """reads a mesh data with index,centroid,x,y
        creates a DF and Points for geometry to return a
        """
        area = pd.read_csv(path, index_col=0)
        area = gpd.GeoDataFrame(
            area, geometry=gpd.points_from_xy(x=area["X"], y=area["Y"]), crs="EPSG:4326"
        )
        area.drop(columns=["centroid", "X", "Y"], inplace=True)
        return area

    def intersect_data_mesh(self, df, mesh):
        mesh["MESH4_ID"] = list(mesh["MESH4_ID"].values.astype("int64"))
        dfa = df[df["area"].isin(list(mesh["MESH4_ID"].values))]
        dfa = dfa.merge(mesh, left_on="area", right_on="MESH4_ID")
        # dfa.drop(columns=["MESH4_ID"], inplace=True)
        gdf = gpd.GeoDataFrame(dfa, geometry="geometry")
        return gdf

    def create_dict_area(
        self, dfd, date_idx, n_hours, area, save=True, name="dict_area"
    ):
        df = {}
        for key in date_idx:
            for h in range(n_hours):
                new_key = key + f"{h:02}"
                df[new_key] = self.intersect_data_mesh(dfd[key][h], area)
                if save:
                    self.store_dict(df, f"{name}")
        return df

    def create_array_one_meshgrid(
        self, dfd, meshcode, date_idx, n_hours, save=False, name="one_mesh"
    ):
        # In mesh : rows = days, columns = hours
        mesh = np.zeros((len(date_idx), n_hours), dtype=np.int64)
        # In mesh : rows = hours, columns = days
        mesh_t = np.zeros((n_hours, len(date_idx)), dtype=np.int64)
        for i, key in enumerate(date_idx):
            for h in range(n_hours):
                new_key = key + f"{h:02}"
                df = dfd[new_key]
                tempdf = df[df.area == meshcode]
                if tempdf.empty:
                    mesh[i][h] = 0
                    mesh_t[h][i] = 0
                else:
                    mesh[i][h] = tempdf["population"].to_list()[0]
                    mesh_t[h][i] = tempdf["population"].to_list()[0]
        if save:
            np.savetxt(
                Path(self.folderpath, "data", f"{self.fpfx}_{name}.csv"),
                mesh,
                delimiter=",",
            )
            np.savetxt(
                Path(self.folderpath, "data", f"{self.fpfx}_{name}_t.csv"),
                mesh_t,
                delimiter=",",
            )
        return mesh, mesh_t

    def meshgrid_array_to_dataframe(self, mesh, date_idx, n_hours=24):
        mesh_1 = pd.DataFrame(
            mesh, columns=np.linspace(0, n_hours - 1, n_hours).astype("int")
        )
        mesh_1.set_index(pd.Index(date_idx), inplace=True)
        mesh_1.index = pd.to_datetime(mesh_1.index)
        return mesh_1

    def bounding_box(self):
        # calculate boundaries
        bbox = np.zeros((len(self.gdfp.keys()), 4))
        for i, key in enumerate(self.gdfp.keys()):
            bbox[i, :] = self.gdfp[str(key)].total_bounds

        self.xmin = bbox[:, 0].min()
        self.ymin = bbox[:, 1].min()
        self.xmax = bbox[:, 2].max()
        self.ymax = bbox[:, 3].max()

        self.bbox = {self.xmin, self.xmax, self.ymin, self.ymax}
        return

    def _max_min_pop_period(self):
        _pmax = np.nan
        _pmin = np.nan
        self.pmax = 0
        self.pmin = 10**10
        for i, key in enumerate(self.gdfp.keys()):
            _pmax = self.gdfp[str(key)].population.max()
            _pmin = self.gdfp[str(key)].population.min()
        if _pmax > self.pmax:
            self.pmax = _pmax
        if _pmin < self.pmin:
            self.pmin = _pmin
        return

    def plot_population(self, ftype=0, save=True):
        if self.pop.empty:
            print("Getting population...")
            self.get_population(ftype)
        plt.close()
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(self.pop.population)
        # p_min_x = self.pop.index(min(self.pop))
        p_min_x = self.pop[["population"]].idxmin().item()
        # p_max_x = self.pop.index(max(self.pop))
        p_max_x = self.pop[["population"]].idxmax().item()
        p_min_y = self.pop.population.min()  # min(self.pop)
        p_max_y = self.pop.population.max()  # max(self.pop)
        ymin, ymax = ax.get_ylim()
        ax.vlines(self.mdate, ymin, ymax, colors="grey", linestyles="dotted")
        ax.scatter([p_min_x, p_max_x], [p_min_y, p_max_y], c="r")
        ax.annotate(
            p_min_y, (p_min_x, p_min_y), (p_min_x + timedelta(hours=0.2), p_min_y + 5)
        )
        ax.annotate(
            p_max_y, (p_max_x, p_max_y), (p_max_x + timedelta(hours=0.2), p_max_y + 5)
        )
        ax.annotate(
            self.mdate,
            (self.mdate, p_max_y),
            (self.mdate + timedelta(hours=0.2), p_max_y + 5),
        )
        ax.set_xlabel("Days")
        ax.set_ylabel("Population")
        ax.set_title(f"Aggregated Population at {self.e_name}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        # trans = ax.get_xaxis_transform()  # x in data untis, y in axes fraction
        # ann = ax.annotate(f"{self.sdate} ~ {self.edate}", xy=(0, -0.1), xycoords=trans)
        if save:
            plt.savefig(
                Path(self.folderpath, "plots", f"{self.fpfx}_pop_ftype{ftype}.png"),
                dpi=300,
            )
        return

    def plot_gdf(self, gdf, cmap="cividis_r", save=True):
        plt.close()
        fig, ax = plt.subplots(1, 1)
        gdf.plot(
            ax=ax,
            # aspect="equal",
            column="population",
            # marker="s",
            # markersize=45,
            cmap=cmap,
            vmin=self.pmin,
            vmax=self.pmax,  # maximum as Tokyo density 6,158 pers/km2 <> 1,500 pers./(500x500)m2
            # figsize=(10, 10),
            alpha=0.6,
            legend=True,
            legend_kwds={"label": "Population"},
        )
        ctx.add_basemap(
            ax,
            zoom=15,
            crs=gdf.crs.to_string(),
            source=ctx.providers.Esri.WorldImagery,
            attribution=False,
        )
        ax.ticklabel_format(useOffset=False, style="plain")
        plt.title(f'{gdf["date"][0]} {int(gdf["time"][0]/100):02d}:00')
        if len(self.bbox) == 0:
            self.bounding_box()
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        if save:
            plt.savefig(
                Path(
                    self.folderpath,
                    "figures",
                    f'{self.fpfx}_{gdf["date"][0]}{gdf["time"][0]:04}.png',
                ),
                dpi=300,
                bbox_inches="tight",
            )
        return

    def plot_aoi(self, gdf, save=False):
        plt.close()
        fig, ax = plt.subplots(1, 1)
        gdf.plot(
            ax=ax,
            column="population",
            alpha=0.0,
        )
        ctx.add_basemap(
            ax,
            zoom=15,
            crs=gdf.crs.to_string(),
            source=ctx.providers.Esri.WorldImagery,
            attribution=False,
        )
        ax.ticklabel_format(useOffset=False, style="plain")
        plt.title(f"{self.e_name}")
        if len(self.bbox) == 0:
            self.bounding_box()
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        if save:
            plt.savefig(
                Path(self.folderpath, "plots", f"{self.fpfx}_{self.aoi_name}.png"),
                dpi=300,
                bbox_inches="tight",
            )
        return

    def make_video_(
        self,
        image_foldername="figures",
        video_name="video",
        fps=1,
        verbose=False,
    ):
        image_folder = Path(self.folderpath, image_foldername)
        video_name = str(Path(self.folderpath, f"{video_name}.mp4"))
        images = [
            img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")
        ]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(
            video_name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        for image in images:
            if verbose:
                print(image)
            video.write(cv2.imread(os.path.join(image_folder, image)))
        cv2.destroyAllWindows()
        video.release()
        return

    # better this one?
    def make_video(self, framerate=5):
        cwd = os.getcwd()
        os.chdir(Path(self.folderpath, "figures"))
        os.system(
            f'ffmpeg -framerate {framerate} -pattern_type glob -i "*.png" -s 3840x2160 -pix_fmt yuv420p ../{self.fpfx}_map.mp4'
        )
        os.chdir(cwd)
        return
    
    def make_video_mesh(self, meshid, framerate=5):
        cwd = os.getcwd()
        os.chdir(Path(self.folderpath, "figures", str(meshid)))
        os.system(
            f'ffmpeg -framerate {framerate} -pattern_type glob -i "*.png" -s 3840x2160 -pix_fmt yuv420p ../{self.fpfx}_{meshid}_map.mp4'
        )
        os.chdir(cwd)
        return
