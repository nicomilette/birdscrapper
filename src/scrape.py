import requests
import csv
import time
import os

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
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists

    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)

def main():
    query = "cnt:\"United States\""  # Adjust this query as needed
    all_recordings = []
    page = 1
    rate_limit = 1  # seconds between requests
    max_pages = None  # Initialize max_pages to None

    while True:
        try:
            print(f"Fetching page {page} with query '{query}'...")
            data = fetch_recordings(query, page)

            if max_pages is None:
                max_pages = int(data.get('numPages', 0))  # Update max_pages based on the first response
                if max_pages == 0:
                    print("No recordings found.")
                    break

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
            if page > max_pages:
                print("All available pages have been fetched.")
                break

            time.sleep(rate_limit)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            break

    # Get the current script directory and construct the full path to the CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(current_dir, '..', 'tables', 'bird_recordings.csv')

    save_to_csv(all_recordings, output_file_path)
    print(f"Data saved to {output_file_path}")

if __name__ == "__main__":
    main()
