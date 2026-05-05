import time

from PIL import Image


def metainfo_collect(IMAGE_SOURCE):
    try:
        img = Image.open(IMAGE_SOURCE)
        exif_data = img._getexif()
        if exif_data is not None:
            gps_info = exif_data.get(34853)  # GPSInfo tag
            if gps_info is not None:
                lat = gps_info.get(2)  # Latitude
                lon = gps_info.get(4)  # Longitude
                timestamp = exif_data.get(36867)  # DateTimeOriginal

                flat = format_coords(lat)

                flon = format_coords(lon)

                formatted_timestamp = time.strftime("%d/%m/%Y %H:%M:%S")
                if timestamp:
                    try:
                        formatted_timestamp = timestamp.replace(':', '-', 2)
                    except Exception as e:
                        print(f"Error formateando fecha: {e}")
                        formatted_timestamp = time.strftime("%d/%m/%Y %H:%M:%S")

                return flat, flon, formatted_timestamp
    except Exception as e:
        print(f"Error al extraer metainformación: {e}")
    return None, None, time.strftime("%d/%m/%Y %H:%M:%S")

def degrees_to_decimal(dms_tuple):
    degrees, minutes, seconds = dms_tuple
    return float(degrees + (minutes / 60.0) + (seconds / 3600.0))

def format_coords(l):
    if l is None:
        f = None
    elif isinstance(l, (tuple, list)):
        try:
            parts = []
            for v in l:
                if isinstance(v, tuple) and len(v) == 2 and v[1] != 0:
                    parts.append(v[0] / v[1])
                else:
                    parts.append(float(v))
            f = degrees_to_decimal(tuple(parts))
        except Exception:
            f = None
    else:
        try:
            f = float(l)
        except Exception:
            f = None
    return f