## Decode the midi from polyvae reconstruction
import numpy as np
import os
from loader.dataloader import MIDI_Render

def vae_gen(data_path, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    mr = MIDI_Render("Irish", 0.15)

    data = np.load(data_path,allow_pickle = True)

    for i,d in enumerate(data):
        # print(d)
        o_note = d["gd"]
        r_note = d["pred"]
        acc = d["acc"]
        dir_path = os.path.join(output_dir, str(i) + "_" + str(acc))
        if os.path.exists(dir_path):
            os.removedirs(dir_path)
        os.mkdir(dir_path)
        for j in range(len(o_note)):
            gd = o_note[j]
            pred = r_note[j]
            mr.data2midi(data = {"notes":gd}, output = os.path.join(dir_path, str(j) + "_o.mid"))
            mr.data2midi(data = {"notes":pred}, output = os.path.join(dir_path, str(j) + "_r.mid"))
        print(i)

def inter_gen(data_path, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    mr = MIDI_Render("Irish", 0.15)

    data = np.load(data_path, allow_pickle = True)
    data = data.item()
    a_note = data["preda"][0]
    b_note = data["predb"][0]
    inters = data["inter"]
    print(a_note)
    print(inters)
    
    mr.data2midi(data = {"notes":a_note}, output = os.path.join(output_dir, "preda.mid"))
    mr.data2midi(data = {"notes":b_note}, output = os.path.join(output_dir, "predb.mid"))

    for i in range(8):
        mr.data2midi(data = {"notes":inters[i][0]}, output = os.path.join(output_dir, str(i) + ".mid"))
        # dir_path = os.path.join(output_dir, str(i) + "_" + str(acc))
        # if os.path.exists(dir_path):
        #     os.removedirs(dir_path)
        # os.mkdir(dir_path)
        # for j in range(len(o_note)):
        #     gd = o_note[j]
        #     pred = r_note[j]
        #     mr.data2midi(data = {"notes":gd}, output = os.path.join(dir_path, str(j) + "_o.mid"))
        #     mr.data2midi(data = {"notes":pred}, output = os.path.join(dir_path, str(j) + "_r.mid"))
        # print(i)


# vae_gen("./reconstruction_2.npy", "./reconstruction_2")
inter_gen("./interpolation.npy","interpolation")