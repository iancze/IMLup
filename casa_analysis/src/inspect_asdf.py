import asdf
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Load an asdf file and print info."
    )
    parser.add_argument("asdf", help="Path to asdf file.")
    args = parser.parse_args()
        
    with asdf.open(args.asdf) as af:
        print(af.info())


if __name__=="__main__":
    main()