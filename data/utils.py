import gzip
import shutil
from pathlib import Path


def unzip_csv_gz_to_directory(input_dir, output_dir, delete_original=False):
    '''Unzip all .csv.gz files from input_dir into output_dir.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing .csv.gz files to unzip.
    output_dir : str or Path
        Destination directory for the unzipped CSV files.
    delete_original : bool, optional
        If True, delete each .gz file after it has been unzipped.

    Returns
    -------
    list of str
        Paths to the successfully unzipped files.
    '''
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    gz_files = list(input_path.glob('*.csv.gz'))

    if not gz_files:
        print(f'No .csv.gz files found in {input_path}')
        return []

    print(f'Found {len(gz_files)} .csv.gz files in {input_path}')
    print(f'Unzipping to {output_path}')
    print(f'{'=' * 50}\n')

    unzipped_files = []

    for gz_file in gz_files:
        output_file = output_path / gz_file.stem

        try:
            with gzip.open(gz_file, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            unzipped_files.append(str(output_file))
            print(f'✓ Unzipped: {gz_file.name}')

            if delete_original:
                gz_file.unlink()
                print(f'  Deleted original: {gz_file.name}')

        except Exception as e:
            print(f'✗ Error unzipping {gz_file.name}: {e}')

    print(f'\n{'=' * 50}')
    print(f'Summary:')
    print(f'  Successfully unzipped: {len(unzipped_files)} files')
    print(f'  Output location: {output_path}')
    print(f'{'=' * 50}')

    return unzipped_files
