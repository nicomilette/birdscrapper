import requests
import csv
import time

def fetch_recordings(query, page=1):
    url = f"https://xeno-canto.org/api/2/recordings?query={query}&page={page}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def save_to_csv(data, filename):
    if not data:
        print("No data to save.")
        return

    keys = data[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)

def main():
    query = "cnt:\"United States\""  # You can change this query to something else, e.g., "cnt:brazil" or a specific species
    all_recordings = []
    page = 1
    rate_limit = 1  # seconds between requests

    while True:
        try:
            print(f"Fetching page {page} with query '{query}'...")
            data = fetch_recordings(query, page)
            print(f"Response for page {page}: {data}")  # Log the response for debugging

            recordings = data.get('recordings', [])
            if not recordings:
                print("No more recordings found.")
                break

            for recording in recordings:
                bird_data = {
                    "id": recording.get("id", ""),
                    "scientific_name": f"{recording.get('gen', '')} {recording.get('sp', '')}",
                    "common_name": recording.get("en", ""),
                    "recording_url": f"https://xeno-canto.org/{recording.get('id', '')}/download",
                    "latitude": recording.get("lat", ""),
                    "longitude": recording.get("lng", ""),
                    "location": recording.get("loc", ""),
                    "date": recording.get("date", ""),
                    "time": recording.get("time", ""),
                    "recordist": recording.get("rec", ""),
                }
                all_recordings.append(bird_data)
            
            page += 1
            time.sleep(rate_limit)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            break

    save_to_csv(all_recordings, 'bird_recordings.csv')
    print("Data saved to bird_recordings.csv")

if __name__ == "__main__":
    main()
