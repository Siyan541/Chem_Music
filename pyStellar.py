import os
from tqdm import tqdm
from multiprocessing import Pool
from AstroChoirGenerator import AstroChoirGenerator, EnhancedAstroChoirGenerator
from AstroMusicGenerator import AstroMusicGenerator, EnhancedAstroMusicGenerator


# --------------------------
# Batch processing module
# --------------------------


def generate_choir(fits_files):
    """Generate stellar choir"""
    full_paths = [os.path.join('/superNova/venv/file/summerSky997', f) for f in fits_files]
    print(f"Full paths for debuggin: {full_paths}")
    
    choir_gen = AstroChoirGenerator(full_paths)
    try:
        print("Choir generating...")
        score = choir_gen.generate_choir()
        output_path = os.path.join('/superNova/venv/file/choirSummer', 'stellar_choir0.mid')
        print(f"Prepare writing the file to: {output_path}")
        score.write('midi', output_path)
        # Generate symphony of a thousand stars
        choir_gen = EnhancedAstroChoirGenerator(full_paths) 
        print("Thousand stars choir generating...")
        # Parameter 997 is an example, adjust as needed
        massive_score = choir_gen.generate_massive_choir(997)
        massive_score.write('midi', 'galaxy_symphony.mid')
        print(f"Choir generated successfully to: {output_path}")
    except Exception as e:
        print(f"Fail to generate choir: {str(e)}")

def process_single(fits_file):
    """Single file processing"""
    try:
        full_path = os.path.join('/superNova/venv/file/summerSky997', fits_file) 
        gen = AstroMusicGenerator(full_path)
        score = gen.generate_full_composition()
        output_path = os.path.join('/superNova/venv/file/midi', fits_file.replace('.fits', '.mid'))
        score.write('midi', output_path)
    except Exception as e:
        print(f"Process {fits_file} Fail: {str(e)}")


if __name__ == "__main__":
    # Create output directories
    os.makedirs('/superNova/venv/file/midi', exist_ok=True)
    os.makedirs('/superNova/venv/file/choirSummer', exist_ok=True)

    # Single-process debug mode (test individual files first)
    # for f in tqdm(fits_files[:1]):
    #     process_single(f)
    
    # Get all spectral files
    fits_files = [f for f in os.listdir('/superNova/venv/file/summerSky997') if f.endswith('.fits')]


    # Multi-process formal processing
    with Pool(8) as pool:
        try:
            list(tqdm(pool.imap(process_single, fits_files), total=len(fits_files)))
        except KeyboardInterrupt:
            print("User interrupted. Exiting...") 


    # Phase 2: Synthesize choir
    generate_choir(fits_files[:997])  # Select 997 stars to generate choir


