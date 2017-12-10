from surprise import Prediction

def reformat_dataset(path_dataset, path_formatted):
    """ Reformat dataset to easily read it with surprise library"""

    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return ('{r},{c},{rat}\n'.format(r=row.strip(), c=col.strip(), rat=rating.strip()))

    # read text file from path
    fp = open(path_dataset, "r")
    data = fp.read().splitlines()[1:]
    fp.close()

    data = [deal_line(line) for line in data]
    
    fp = open(path_formatted, 'w+')
    fp.writelines(data)
    fp.close()
    

def export_predictions(prediction):
    """ Export prediction to kaggle format"""
    # Store users, items and ratings in three arrays
    
    N = len(prediction)
    users = []
    items = []
    rat = []
    
    for j, pred in enumerate(prediction):
        users.append(pred.uid)
        items.append(pred.iid)
        rat.append(pred.est)
        
    # Format preditions in the kaggle format
    data = []
    for j in range(N):
        data.append('r{u}_c{i},{r}\n'.format(u=users[j], i=items[j], r = rat[j]))
        
    # Write predictions in a csv file
    fp = open('./final_prediction.csv', 'w')
    fp.writelines(data)
    fp.close()