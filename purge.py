from pymongo import MongoClient
import click



@click.command()
@click.option("--folder", default=False)
def main(folder):
    if not folder:
        print("[PURGE] You need to specify the table/folder name")
    else:
        ## Init the client
        client = MongoClient()
        collection = client.superGo[str(folder)]

        bulk = collection.initialize_unordered_bulk_op()

        ## Remove 90% of the games
        total_games = collection.find().count()
        new_games = int(total_games * 0.90)
        collection.remove({"id": {"$lte": new_games}})

        ## Update the ids
        games = collection.find()
        new_id = 0
        for game in games:
            bulk.find({'id': game['id']}).update({'$set': {"id": new_id}})
            new_id += 1
        bulk.execute()

        

if __name__ == "__main__":
    main()