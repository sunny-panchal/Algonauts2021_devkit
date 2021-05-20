import os

from perform_encoding import build_argparser


def main(args):
    track = args.track

    if track == 'full_track':
        ROIs = ['WB']
    else:
        ROIs = ['LOC', 'FFA', 'STS', 'EBA', 'PPA', 'V1', 'V2', 'V3', 'V4']

    for roi in ROIs:
        for sub in range(1, 11):
            args.roi = roi  # replace default
            args.sub = sub  # replace default
            cmd_string = f'python perform_encoding.py'

            for arg in vars(args):
                if arg is not 'track':
                    cmd_string += f' --{arg} {vars(args)[arg]}'

            print("Starting ROI: ", roi, "sub: ", sub)
            os.system(cmd_string)
            print("Completed ROI: ", roi, "sub: ", sub)
            print("----------------------------------------------------------------------------\n")


if __name__ == "__main__":
    # build the argparser from perform_encoding.py
    parser = build_argparser()
    parser.description = 'Generates predictions for all subs all ROIs for a given track'

    parser.add_argument('-t', '--track',
                        help='mini_track for all ROIs, full_track for whole brain (WB)',
                        default='mini_track',
                        type=str)

    main(parser.parse_args())
