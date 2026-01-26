import test
import time
for n in range(1, 16):
    test.main([n, "output.kml", "all", True])
    time.sleep(1)