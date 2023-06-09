import os
import shutil

if __name__ == "__main__":

    cities = {
        "manaus": "INMET_N_AM_A101",
        "teresina": "INMET_NE_PI_A312",
        "rio_janeiro": "INMET_SE_RJ_A652",
        "brasilia": "INMET_CO_DF_A001",
        "curitiba": "INMET_S_PR_A807",
    }

    source_folder = r"C:\Users\iflr\Documents\inmet"
    destination_folder = "files_v2"

    os.chdir(source_folder)

    for dirpath, dirnames, filenames in os.walk(source_folder):

        if dirpath == source_folder:
            continue

        for filename in filenames:
            for city, station in cities.items():
                if station in filename:
                    # Build the full path to the source file
                    source_file = os.path.join(dirpath, filename)

                    # Build the full path to the destination file
                    destination_file = os.path.join(destination_folder, filename)

                    station_code = station.split("_")[-1]
                    destination_full = f"{destination_folder}/{city}_{station_code.lower()}"
                    if not os.path.exists(destination_full):
                        os.makedirs(destination_full)

                    destination_file = os.path.join(destination_full, filename.lower())
                    shutil.copyfile(source_file, destination_file)