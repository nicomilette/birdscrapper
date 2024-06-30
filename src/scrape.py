import requests
import csv
import time
import os
import config

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

            # Calculate and print progress percentage
            progress_percentage = (page / max_pages) * 100
            print(f"{page}/{max_pages} pages fetched\n({progress_percentage:.2f}%)")

            time.sleep(rate_limit)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            break
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, config.CSV_UNFILTERED)
    save_to_csv(all_recordings, file_path)
    print(f"Data saved to {config.CSV_UNFILTERED}")

if __name__ == "__main__":
    main()
