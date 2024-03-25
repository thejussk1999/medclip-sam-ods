import sys,os,cv2,tqdm
import random as rand

import ImageSegmentation as seg
import RegionOperator as ro
import SimilarityOperator as so

import pandas as pd

def generate_image(img, rgset):
    random_color = lambda: (int(rand.random() * 255), int(rand.random() * 255), int(rand.random() * 255))
    color = [random_color() for i in range(len(rgset))]

    save_img = img.copy()
    bounding_boxes=[]
    for i in range(len(rgset)):
        bounding_boxes.append({'x_min':rgset[i].rect.left,'x_max':rgset[i].rect.top,'y_min':rgset[i].rect.right,'y_max':rgset[i].rect.bottom})
        cv2.rectangle(save_img, (rgset[i].rect.left, rgset[i].rect.top), (rgset[i].rect.right, rgset[i].rect.bottom), color[i], 2)

    return save_img, bounding_boxes

def main(sigma,k,min_size,smallest,largest,distortion,img,bbox_file):
    img_temp=cv2.imread(img)
    H,W,_=img_temp.shape
    max_HW=max(H,W)
    if max_HW>1000:#
        factor=(max_HW//512)+1
        H//=factor
        W//=factor
        print(factor,H,W)
    img=cv2.resize(img_temp,(W,H))

    ufset = seg.segment_image(sigma, k, min_size, img)

    rgset = ro.extract_region(img, ufset)
    nbset = ro.extract_neighbour(rgset)

    height, width, channel = img.shape
    im_size = width * height
    simset = so.calc_init_similarity(rgset, nbset, im_size)

    while simset != []:
        sim_value = lambda element: element[0]
        max_sim_idx = simset.index(max(simset, key=sim_value))

        a = simset[max_sim_idx][1]
        b = simset[max_sim_idx][2]

        rgset.append(ro.merge_region(rgset[a], rgset[b]))

        for sim in simset:
            if sim[1] == a or sim[1] == b or sim[2] == a or sim[2] == b:
                simset.remove(sim)

                if (sim[1] == a and sim[2] == b) or (sim[1] == b and sim[2] == a):
                    continue

                new_a = rgset.index(rgset[-1])
                new_b = sim[2] if sim[1] == a or sim[1] == b else sim[1]
                simset.append((so.calc_similarity(rgset[new_a], rgset[new_b], im_size), new_a, new_b))

    proposal = []

    for rg in rgset:
        if ro.has_same_rect_region(proposal, rg.rect):
            continue
        if rg.size < smallest or rg.size > largest:
            continue
        rg_wh = rg.rect.get_width() / rg.rect.get_height()
        rg_hw = rg.rect.get_height() / rg.rect.get_width()
        if rg_wh > distortion or rg_hw > distortion:
            continue

        proposal.append(rg)

    save_img, bounding_boxes = generate_image(img, proposal)
    # for i in bounding_boxes:
    #     print(i)
    cv2.imwrite(bbox_file, save_img)
    return bounding_boxes

if __name__ == '__main__':
    # min_limit => Starting index from the folder
    # max_limit => Ending index from the folder
    min_limit,max_limit=int(sys.argv[1]),int(sys.argv[2])

    df=pd.DataFrame([])
    sigma=0.5 
    k=400 
    min_size=50 
    smallest=100 
    largest=10000 
    distortion=1.25

    files=os.listdir('./archive/test/')
    imgs=[i for i in files if i.endswith(".png")]
    for img in tqdm.tqdm(imgs[min_limit:max_limit]):
        bbox_file='./archive/bboxed/'+img.split('.')[0]+'_bbox.png'
        img='./archive/test/'+img
        bbox=main(sigma,k,min_size,smallest,largest,distortion,img,bbox_file)
        df = pd.concat([df,pd.DataFrame([{'File':img,'Bounding box':bbox}])])
        # df.to_csv('bbox_final_test_{}_{}.csv'.format(min_limit,max_limit),index=False)
