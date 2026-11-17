import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from geolifeclef.raster_bridge import nearest_valid_value


def test_nearest_valid_fill_respects_circular_radius_and_edges():
    data = np.full((11, 11), -9999, dtype='float32')
    data[3, 4] = 17
    with MemoryFile() as memory:
        with memory.open(driver='GTiff', width=11, height=11, count=1, dtype='float32', nodata=-9999,
                         transform=from_origin(0, 20, 1, 1)) as src:
            src.write(data, 1)
            assert np.isnan(nearest_valid_value(src, 0, 0, maximum_radius=4)[0])
            value, distance, _ = nearest_valid_value(src, 0, 0, maximum_radius=5)
            assert value == 17 and distance == 5
            assert nearest_valid_value(src, 3, 5, maximum_radius=2)[0] == 17
