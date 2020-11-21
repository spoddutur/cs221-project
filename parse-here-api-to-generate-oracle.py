import requests
access_token = "<>HERE_API_ACCESS_TOKEN"
head = {'Authorization': 'Bearer ' + access_token}
url0="https://geocoder.ls.hereapi.com/search/6.2/geocode.json?languages=en-US&maxresults=1&searchtext=CHEZ ROUX,ESTRABLIN,38780,France"
url1 = "https://geocoder.ls.hereapi.com/6.2/geocode.json?searchtext=72+AVENUE+DE+LA+GARE+CHEZ+SARL+BORA+CAGNES+SUR+MER+France+06800&additionaldata=IncludeMicroPointAddresses,true&locationattributes=mapReference&gen=9"
url = "https://geocoder.ls.hereapi.com/6.2/geocode.json?languages=en-US&searchtext=72+AVENUE+DE+LA+GARE+CHEZ+SARL+BORA+CAGNES+SUR+MER+France+06800"

def getHereApiResponse(searchtext):
    base_url = "https://geocoder.ls.hereapi.com/6.2/geocode.json?languages=en-US&searchtext="
    url = base_url + searchtext
    response = requests.get(url, headers=head)    
    jsonResponse = response.json()
    # print(jsonResponse)
    has_response = "Response" in jsonResponse
    if has_response:
        has_view = "View" in jsonResponse["Response"]
        if has_view:
            views = jsonResponse["Response"]["View"]
            if len(views) > 0:
                results = jsonResponse["Response"]["View"][0]["Result"]
                if len(results) > 0:
                    return results[0]["Location"]["Address"]
    return ""

# READ INPUT DATA
INPUT_FILE_PATH = "/Users/sruthip/Downloads/part0-parsed-combined-street1.csv"
df = pd.read_csv(INPUT_FILE_PATH, header=0)[["id", "model_input"]]

out = []
start = timenow()
for i in range(len(df)):
    token = df.loc[i, "token"]
    address = df.loc[i, "model_input"]
    id = df.loc[i, "id"]

    # write every 1000 files
    if i>1 and i%1000 == 0:
        print(i, start, timenow())
        out_df = pd.DataFrame(out)
        out_df.to_csv("/Users/sruthip/Downloads/data/hereapiresponse/part0-parsed-combined-street1lakhlabels-"+str(i)+".csv", index=False)
        out=[]
        start = timenow()
    here_response = getHereApiResponse(address)
    out.append((token, id, address, here_response))

# Writing last batch
i = i+1
out_df = pd.DataFrame(out)
out_df.to_csv("/Users/sruthip/Downloads/data/hereapiresponse/part0-parsed-combined-street1lakhlabels-"+str(i)+".csv", index=False)
        
