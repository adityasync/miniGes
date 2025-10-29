import os
root='dataset'
count=0
for category in sorted(os.listdir(root)):
    cpath=os.path.join(root, category)
    if not os.path.isdir(cpath):
        continue
    for label in sorted(os.listdir(cpath)):
        if os.path.isdir(os.path.join(cpath, label)):
            count += 1
print(count)
