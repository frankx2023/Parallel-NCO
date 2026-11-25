import sys
from reproduce_inference import main

if __name__ == '__main__':
    sys.argv.extend(['--problems', 'MC'])
    main()
