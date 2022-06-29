import csv
import praw
import creds

# Authorization
reddit = praw.Reddit(client_id = creds.CLIENT_ID, client_secret = creds.SECRET_TOKEN, user_agent = 'my user agent', username = creds.USERNAME, password = creds.PASSWORD)

# Assessing a Subreddit - r/singapore
subreddit = reddit.subreddit('singapore')

# Keep count of entries
entry_count = 0
post_count = 0

# Write to csv file
csv_file = open('raw_data.csv', 'w', encoding="utf-8")

# Assessing Submissions
for submission in subreddit.hot(limit=None):
    # Write the post's title and selftext as an entry each
    csv_file.write(submission.title + "\n")
    entry_count += 1
    post_count += 1

    #csv_file.write(submission.selftext)     ## removed because there were alot of links and pretty text
    
    all_comments = submission.comments.list()

    print(f"post...{post_count}, writing...{entry_count}")

    for c in all_comments:
        if hasattr(c, 'body'):
            entry_count += 1
            csv_file.write(c.body + "\n")
            
# Close csv file
csv_file.close()

# Assessing a Subreddit - r/singaporeraw
subreddit = reddit.subreddit('singaporeraw')

# Write to csv file
csv_file = open('raw_data.csv', 'a', encoding="utf-8")

# Assessing Submissions
for submission in subreddit.hot(limit=None):
    # Write the post's title and selftext as an entry each
    csv_file.write(submission.title + "\n")
    entry_count += 1
    post_count += 1

    #csv_file.write(submission.selftext)     ## removed because there were alot of links and pretty text
    
    all_comments = submission.comments.list()

    print(f"post...{post_count}, writing...{entry_count}")

    for c in all_comments:
        if hasattr(c, 'body'):
            entry_count += 1
            csv_file.write(c.body + "\n")

print(f"Entry Count: {entry_count}")

# Close csv file
csv_file.close()