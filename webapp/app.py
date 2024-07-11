import os
import logging
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, '../bird_recordings')
SPECTROGRAM_DIR = os.path.join(BASE_DIR, '../spectrograms')
SELECTED_FILE = os.path.join(BASE_DIR, 'selected.txt')
DISCARDED_FILE = os.path.join(BASE_DIR, 'discarded.txt')

def init_files():
    if not os.path.exists(SELECTED_FILE):
        open(SELECTED_FILE, 'w').close()
    if not os.path.exists(DISCARDED_FILE):
        open(DISCARDED_FILE, 'w').close()

init_files()

def get_species_list():
    species_set = set()
    for file_name in os.listdir(AUDIO_DIR):
        if file_name.endswith('.mp3'):
            species = '_'.join(file_name.split('_')[:-1])
            species_set.add(species)
    return sorted(list(species_set))

def get_species_stats(species):
    selected_count = 0
    discarded_count = 0

    with open(SELECTED_FILE, 'r') as sf:
        selected_count = sum(1 for line in sf if line.startswith(species))

    with open(DISCARDED_FILE, 'r') as df:
        discarded_count = sum(1 for line in df if line.startswith(species))

    return {'selected': selected_count, 'discarded': discarded_count}

def get_filtered_audio_files(species):
    selected_files = set()
    discarded_files = set()

    with open(SELECTED_FILE, 'r') as sf:
        for line in sf:
            if line.startswith(species):
                selected_files.add(line.strip())

    with open(DISCARDED_FILE, 'r') as df:
        for line in df:
            if line.startswith(species):
                discarded_files.add(line.strip())

    all_files = [f for f in os.listdir(AUDIO_DIR) if f.startswith(species) and f.endswith('.mp3')]
    filtered_files = [f for f in all_files if f not in selected_files and f not in discarded_files]

    return filtered_files

@app.route('/')
def index():
    species = get_species_list()
    species_stats = {sp: get_species_stats(sp) for sp in species}
    return render_template('index.html', species_stats=species_stats)

@app.route('/audio/<species>')
def get_audio_files(species):
    audio_files = get_filtered_audio_files(species)
    audio_data = [{'audio_file': audio_file, 'audio_path': f'/audio_files/{audio_file}'} for audio_file in audio_files]
    return jsonify(audio_data)

@app.route('/audio_files/<path:filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)

@app.route('/spectrograms/<path:filename>')
def serve_spectrogram(filename):
    return send_from_directory(SPECTROGRAM_DIR, filename)

@app.route('/select', methods=['POST'])
def select_recording():
    data = request.json
    species = data['species']
    audio_file = data['audio_file']

    with open(SELECTED_FILE, 'a') as sf:
        if get_species_stats(species)['selected'] < 20:
            sf.write(f"{audio_file}\n")
            return jsonify(success=True)
    return jsonify(success=False, error='Already 20 selected')

@app.route('/discard', methods=['POST'])
def discard_recording():
    data = request.json
    species = data['species']
    audio_file = data['audio_file']

    with open(DISCARDED_FILE, 'a') as df:
        df.write(f"{audio_file}\n")
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True)
