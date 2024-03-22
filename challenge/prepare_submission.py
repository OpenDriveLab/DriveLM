import json

# Please fill in your team information here
method = ""  # <str> -- name of the method
team = ""  # <str> -- name of the team, !!!identical to the Google Form!!!
authors = [""]  # <list> -- list of str, authors
email = ""  # <str> -- e-mail address
institution = ""  # <str> -- institution or company
country = ""  # <str> -- country or region


def main():
    with open('output.json', 'r') as file:
        output_res = json.load(file)

    submission_content = {
        "method": method,
        "team": team,
        "authors": authors,
        "email": email,
        "institution": institution,
        "country": country,
        "results": output_res
    }

    with open('submission.json', 'w') as file:
        json.dump(submission_content, file, indent=4)

if __name__ == "__main__":
    main()
