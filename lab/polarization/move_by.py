"""Rotate the KDC101 rotation mount by a relative angle.

Usage:
    python move_by.py 10        # rotate +10 deg
    python move_by.py -5.5      # rotate -5.5 deg
    python move_by.py           # just print the current angle
"""
import argparse

import thorlabs_apt.thorlabs_apt as apt

ROTATION_MOUNT_SN = 27268833  # KDC101 driving the rotation mount


def main():
    parser = argparse.ArgumentParser(description="Rotate the KDC101 rotation mount by a relative angle.")
    parser.add_argument("degrees", type=float, nargs="?", default=None,
                        help="relative rotation in degrees (omit to only print the current angle)")
    parser.add_argument("--sn", type=int, default=ROTATION_MOUNT_SN, help="controller serial number")
    args = parser.parse_args()

    motor = apt.Motor(args.sn)
    start = motor.position
    print(f"current: {start:.3f} deg")
    if args.degrees is None:
        return

    motor.move_by(args.degrees, blocking=True)
    end = motor.position
    print(f"moved:   {end - start:+.3f} deg -> now at {end:.3f} deg")


if __name__ == "__main__":
    main()
