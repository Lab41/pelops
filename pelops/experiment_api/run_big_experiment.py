from experiment import ExperimentGenerator
from pelops import utils

@timewrapper
def main():
    veri_unzipped_path = ""
    num_cams = 1
    num_cars_per_cam = 10
    drop_percentage = 0
    seed = 11
    time = 5
    typ = 2

    # create the generator
    exp = ExperimentGenerator(veri_unzipped_path, num_cams, num_cars_per_cam, drop_percentage, seed, time, typ)

    for i in range(0, 10000):
        # generate the experiment
        set_num = 1
        print("=" * 80)
        for camset in exp.generate():
            print("Set #{}".format(set_num))
            print("Target car: {}".format(exp.target_car.name))
            print("-" * 80)
            for image in camset:
                print("name: {}".format(image.name))
                """
                print("filepath: {}".format(image.filepath))
                print("type: {}".format(image.type))
                print("car id: {}".format(image.car_id))
                print("camera id: {}".format(image.camera_id))
                print("timestamp: {}".format(image.get_timestamp()))
                print("binary: {}".format(image.binary))
                """
                print("-" * 80)
            print("=" * 80)
            set_num = set_num + 1

    return

if __name__ == '__main__':
    main()