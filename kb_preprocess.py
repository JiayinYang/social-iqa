import csv
import random

obj_name_key = ['people', 'their', 'other', 'person', 'someone', \
                'others', 'they ', 'persony', 'personx', 'the person', 'she ', 'he ', 'the people']


# https://nameberry.com/unisex-names
unisex_names = ['Avery', 'Riley', 'Jordan', 'Angel', 'Parker', 'Sawyer', 
                'Peyton', 'Quinn', 'Blake', 'Hayden', 'Taylor', 'Alexis', 
                'Rowan', 'Charlie', 'Emerson', 'Finely', 'River', 'Ariel',
                'Emery', 'Morgan', 'Elliot', 'London', 'Eden', 'Elliott', 
                'Karter', 'Dakota', 'Reese', 'Zion', 'Remington', 'Payton',
                'Amari', 'Phoenix', 'Kendall', 'Harley', 'Rylan', 'Marley', 
                'Dallas', 'Skyler', 'Spencer', 'Sage', 'Kyrie', 'Lyric', 
                'Ellis', 'Rory', 'Remi', 'Justice', 'Ali', 'Haven', 'Tatum', 'Kamryn']

def format_infer(inf):
    inf = inf.lower().replace("person x", "personx").replace("person y", "persony").replace("person z", "personz") \
                     .replace(" x ", " personx ").replace(" y ", " persony ").replace(" z ", " personz ") \
                     .strip()
    if inf.endswith(" x"):
        inf = inf.replace(" x", " personx") 
    if inf.endswith(" y"):
        inf = inf.replace(" y", " persony")
    if inf.endswith(" z"):
        inf = inf.replace(" z", " personz")
        
    if inf.startswith("x "):
        inf = inf.replace("x ", "personx ") 
    if inf.startswith("y "):
        inf = inf.replace("y ", "persony ")
    if inf.startswith("z "):
        inf = inf.replace("z ", "personz ")
    return inf

def rand_naming(name_list):
    return name_list[random.randint(0, 49)]

# read filling words
filling_words = dict()

with open("v4_atomic_all_agg_completed_words.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if row[1]:
            num = str(int(row[0].strip())+1)
            if num not in filling_words:
                filling_words[num] = []
            filling_words[num].append(row[1].strip())
       
    
# read knowledge graph
event_infer_data_filled = []
num = 0
with open("atomic_data/v4_atomic_all_agg.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if num == 0:
            num +=1 
            continue
        if str(num) in filling_words:
            events = [row[0].lower().replace("___", w) for w in filling_words[str(num)]]
        else:
            events = [row[0].lower()]
        infers = [row[i].strip('][').replace('"', "").split(', ') for i in range(1,11)]
        infers_clear = []
        for i in infers:
            infers_clear.append([x for x in i if x != 'none'])   
        for e in events:
            event_infer_data_filled.append((e,infers_clear))
        num += 1
        

oeffect_sen = []
oreact_sen = []
owant_sen = []
xattr_sen = []
xeffect_sen = []
xintent_sen = []
xneed_sen = []
xreact_sen = []
xwant_sen = []

# connect sentenses
for event,infer in event_infer_data_filled:
    
    oeffects = infer[0]
    for e in oeffects:
        this_obj = ''

        e = format_infer(e)

        # add objects
        if sum([e.startswith(k) for k in obj_name_key]) == 0:
            if "persony" in event:
                this_obj = 'persony '
            elif "personz" in event:
                this_obj = 'personz '
            else:
                this_obj = 'others '

        completed_sen = "if " + event + ", then " + this_obj + e
        oeffect_sen.append(completed_sen)
        
        
    oreacts = infer[1]
    for r in oreacts:
        this_obj = ''

        r = format_infer(r)

        # add objects
        if sum([r.startswith(k) for k in obj_name_key]) == 0:
            if "persony" in event:
                this_obj = 'persony will be '
            elif "personz" in event:
                this_obj = 'personz will be '
            else:
                this_obj = 'others will be '

        completed_sen = "if " + event + ", then " + this_obj + r
        oreact_sen.append(completed_sen)
    
    owants = infer[2]
    for w in owants:
        this_obj = ''

        w = format_infer(w)

        # add objects
        if sum([w.startswith(k) for k in obj_name_key]) == 0:
            if "persony" in event:
                this_obj = 'persony will want '
            elif "personz" in event:
                this_obj = 'personz will want '
            else:
                this_obj = 'others will want '

        completed_sen = "if " + event + ", then " + this_obj + w
        owant_sen.append(completed_sen)
            
    xattrs = infer[3]
    for a in xattrs:
        this_sbj = ''

        a = format_infer(a)

        # add subjects
        if sum([a.startswith(k) for k in obj_name_key]) == 0:
            this_sbj = 'personx is '

        completed_sen = "if " + event + ", then " + this_sbj + a
        xattr_sen.append(completed_sen)
    
    xeffects = infer[4]
    for e in xeffects:
        this_sbj = ''

        e = format_infer(e)

        # add subjects
        if sum([e.startswith(k) for k in obj_name_key]) == 0:
            this_sbj = 'personx '

        completed_sen = "if " + event + ", then " + this_sbj + e
        xeffect_sen.append(completed_sen)
    
    xintents = infer[5]
    for i in xintents:
        this_sbj = ''

        i = format_infer(i)

        # add subjects
        if sum([i.startswith(k) for k in obj_name_key]) == 0:
            this_sbj = 'personx wanted '

        completed_sen = "if " + event + ", then " + this_sbj + i
        xintent_sen.append(completed_sen)
    
    xneeds = infer[6]
    for n in xneeds:
        this_sbj = ''

        n = format_infer(n)

        # add subjects
        if sum([n.startswith(k) for k in obj_name_key]) == 0:
            this_sbj = 'personx needs '

        completed_sen = "if " + event + ", then " + this_sbj + n
        xneed_sen.append(completed_sen)
        
    xreacts = infer[7]
    for r in xreacts:
        this_sbj = ''

        r = format_infer(r)

        # add subjects
        if sum([r.startswith(k) for k in obj_name_key]) == 0:
            this_sbj = 'personx will be '

        completed_sen = "if " + event + ", then " + this_sbj + r
        xreact_sen.append(completed_sen)
        
    xwants = infer[8]
    for w in xwants:
        this_sbj = ''

        w = format_infer(w)

        # add subjects
        if sum([w.startswith(k) for k in obj_name_key]) == 0:
            this_sbj = 'personx will want '

        completed_sen = "if " + event + ", then " + this_sbj + w
        xwant_sen.append(completed_sen)


all_sen = oeffect_sen + oreact_sen + owant_sen + xattr_sen + xeffect_sen + \
          xintent_sen + xneed_sen + xreact_sen + xwant_sen 

# randomly rename personx, persony, personz
random.seed(1)
all_sen_named = []
for sen in all_sen:
    perX = rand_naming(unisex_names)
    perY = ''
    perZ = ''
    if "persony" in sen:
        perY = perX
        while perY == perX:
            perY = rand_naming(unisex_names)
    if "personz" in sen:
        perZ = perX
        while perZ == perX or perZ == perY:
            perZ = rand_naming(unisex_names)
    
    all_sen_named.append(sen.replace("personx", perX).replace("persony", perY).replace("personz", perZ))
    
# write sentenses into csv file
with open('final_sentences.csv', 'w') as output:
    writer = csv.writer(output)
    for sen in all_sen_named:
        writer.writerow([sen])