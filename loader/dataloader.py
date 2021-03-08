# A dataloader for processing polyphonic music
import pretty_midi as pyd
import numpy as np
import os
import random
import copy

                                
class Dataset:
    def __init__(self, filedir = None):
        self.filedir = filedir
        self.midifile = []
        self.max_voice = 10
        print("Dataset Init:" + filedir)
        if filedir is not None:
            for i,f in enumerate(os.listdir(filedir)):
                if f.endswith(".mid"):
                    self.midifile.append(pyd.PrettyMIDI(os.path.join(filedir, f)))
                if (i + 1) % 100 == 0:
                    print("processed:", i)
                    # break
            print("Dataset Loaded!")
        else:
            print("Dataset Folder Invalid!")
    
    def split_dataset(self, r_train = 0.8, r_validate = 0.1, r_test = 0.1):
        newmidi = self.midifile
        random.shuffle(newmidi)
        k = int(len(self.midifile) * r_train)
        v = int(len(self.midifile) * (r_train + r_validate))
        p = int(len(self.midifile) * (r_train + r_validate + r_test))
        self.train = newmidi[:k]
        self.validate = newmidi[k:v]
        self.test = newmidi[v:p]
        print("Finish Spliting!")

    def process_midi(self, midi_file, measure_length = 16, min_step = 0.15):
        measure_time = round(measure_length * min_step,2)
        notes = []
        for ins in midi_file.instruments:
            if ins.name == "PIANO":
                notes += ins.notes
        notes.sort(key = lambda x: x.pitch)
        notes.sort(key = lambda x: x.start)
        time_bias = 0.0
        measures = []
        measure = []
        for note in notes:
            if note.start - time_bias >= measure_time:
                measures.append(measure)
                measure = []
                time_bias += measure_time
                time_bias = round(time_bias,2)
            if 0.0 <= round(note.start - time_bias,2) < measure_time:
                measure.append([100, note.pitch, round(note.start - time_bias,2), round(min(note.end - time_bias, measure_time),2)])
        if len(measure) > 0:
            measures.append(measure)
        return measures

    def note2token(self, notes, measure_length = 16, min_step = 0.15):
        rest_state = 129
        hold_state = 128
        token = [rest_state] * measure_length
        for note in notes:
            sta = int(round(note[2] / min_step))
            end = int(round(note[3] / min_step))
            if sta < measure_length and end <= measure_length:
                token[sta] = note[1]
                token[sta + 1: end] = [hold_state] * (end - sta - 1)
        if len(token) != measure_length:
            print("????")
        return token
            
    def process_measure(self, measure):
        if len(measure) == 0:
            return None
        simu_group = []
        simu_notes = []
        pos = measure[0][2]
        simu_notes.append(measure[0])
        for note in measure[1:]:
            if note[2] != pos:
                simu_group.append(simu_notes)
                pos = note[2]
                simu_notes = []
            if note[2] == pos:
                simu_notes.append(note)
        if len(simu_notes) > 0:
            simu_group.append(simu_notes)
        layer_index = [-1] * self.max_voice 
        layer = []
        # print(simu_group)
        max_v = 0
        for sg in simu_group:
            max_v = max(max_v, len(sg))
        if max_v == 1:
            for i in range(self.max_voice):
                layer_index[i] = [0] * len(simu_group)
        else:
            if max_v > self.max_voice:
                max_v = self.max_voice
            # print(max_v)
            inter_step = self.max_voice / (max_v - 1)
            for i in range(max_v):
                idx = min(self.max_voice - 1, int(round(inter_step * i)) - 1)
                if idx < 0:
                    idx = 0
                layer_index[idx] = []
                for j in range(len(simu_group)):
                    layer_index[idx].append(min(len(simu_group[j]) - 1, i))
            temp_index = [0] * len(simu_group)
            # print(layer_index)
            for i in range(len(layer_index)):
                if layer_index[i] == -1:
                    pos = i
                    # print(i)
                    tail = len(simu_group) - 1 
                    while(pos < len(layer_index) and layer_index[pos] == -1):
                        if tail >= 0:
                            temp_index[tail] = min(len(simu_group[tail]) - 1, temp_index[tail] + 1)
                            tail -= 1
                        layer_index[pos] = temp_index[::]
                        pos = pos + 1
                    # print(layer_index)
                else:
                    temp_index = layer_index[i][::]
        # print(layer_index)
        for i, indexes in enumerate(layer_index):
            layer.append([])
            for j,idx in enumerate(indexes):
                layer[i].append(simu_group[j][idx])
            layer[i] = self.note2token(layer[i])
        return {"raw": measure, "layers": layer}

    def process(self, measure_length = 16, min_step = 0.15, outdir = "layer_output/"):
        print("Measure Time:", measure_length * min_step)
        self.train_measures = []
        self.validate_measures = []
        self.test_measures = []

        self.train_layers = []
        self.validate_layers = []
        self.test_layers = []

        for i in self.train:
            self.train_measures += self.process_midi(i, measure_length, min_step)
        for i in self.validate:
            self.validate_measures += self.process_midi(i, measure_length, min_step)
        for i in self.test:
            self.test_measures += self.process_midi(i, measure_length, min_step)

        for i, m in enumerate(self.train_measures):
            temp = self.process_measure(m)
            if temp is not None:
                self.train_layers.append(temp)
                self.test_layer(inputs = temp, output = outdir + "train_" + str(i) + ".mid")
            if i % 100 == 0:
                print("prcessed:", i)

        for i, m in enumerate(self.validate_measures):
            temp = self.process_measure(m)
            if temp is not None:
                self.validate_layers.append(temp)
                self.test_layer(inputs = temp, output = outdir + "validate_" + str(i) + ".mid")
            if i % 100 == 0:
                print("prcessed:", i)

        for i, m in enumerate(self.test_measures):
            temp = self.process_measure(m)
            if temp is not None:
                self.test_layers.append(temp)
                self.test_layer(inputs = temp, output = outdir + "test_" + str(i) + ".mid")
            if i % 100 == 0:
                print("prcessed:", i)
        return True
    
    def test_layer(self, inputs, output = "test.mid", measure_time = 2.4, min_step = 0.15):
        gen_midi = pyd.PrettyMIDI(initial_tempo = 100)
        melodies = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
        time_bias = 0.0
        measure = inputs["raw"]
        layers = inputs["layers"]
        for note in measure:
            melodies.notes.append(
                pyd.Note(
                    velocity = note[0],
                    pitch = note[1],
                    start = note[2] + time_bias,
                    end = note[3] + time_bias
                )
            )
        time_bias += measure_time
        time_bias = round(time_bias,2)
        rest_state = 129
        hold_state = 128
        for layer in layers:
            prev = 129
            timeline = 0.0
            duration = 0.0
            for token in layer:
                if 0 <= token < hold_state or token == rest_state:
                    if 0 <= prev < hold_state:
                        melodies.notes.append(
                            pyd.Note(
                                velocity = 100,
                                pitch = prev,
                                start = timeline + time_bias,
                                end = timeline + duration + time_bias
                            )
                        )
                    timeline += duration
                    prev = token
                    if token == rest_state:
                        timeline += min_step
                        duration = 0.0
                    else:
                        duration = min_step
                elif token == hold_state:
                    duration += min_step
            if 0 <= prev < hold_state:
                melodies.notes.append(
                    pyd.Note(
                        velocity = 100,
                        pitch = prev,
                        start = timeline + time_bias,
                        end = timeline + duration + time_bias
                    )
                )
            time_bias += measure_time
            time_bias = round(time_bias,2)
        gen_midi.instruments.append(melodies)
        gen_midi.write(output)
    
    def test_out(self, measures, output = "test.mid", measure_time = 2.4):
        gen_midi = pyd.PrettyMIDI(initial_tempo = 100)
        melodies = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
        time_bias = 0.0
        for measure in measures:
            for note in measure:
                melodies.notes.append(
                    pyd.Note(
                        velocity = note[0],
                        pitch = note[1],
                        start = note[2] + time_bias,
                        end = note[3] + time_bias
                    )
                )
            time_bias += measure_time
            time_bias = round(time_bias,2)
        gen_midi.instruments.append(melodies)
        gen_midi.write(output)
        
class MIDI_Render:
    def __init__(self, datasetName, minStep = 0.03125):
        self.datasetName = datasetName
        self.minStep = minStep
    def data2midi(self, data, recogLevel = "Mm", output = "test.mid"):
        gen_midi = pyd.PrettyMIDI(initial_tempo=100.0)
        melodies = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
        chords = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
        if self.datasetName == "Nottingham":
            # 130 one hot vectors 
            # 0-127 for pitch
            # 128 for hold 129 for rest
            rest_pitch = 129
            hold_pitch = 128
            cl = Chord_Loader(recogLevel = recogLevel)
            time_shift = 0.0
            local_duration = 0
            prev = "NC"
            for chord in data["chords"]:
                if chord == "":
                    continue
                chord = cl.index2name(x = int(chord))
                if chord == prev:
                    local_duration += 1
                else:
                    if prev == "NC":
                        prev = chord
                        time_shift += local_duration * self.minStep
                        local_duration = 1
                    else:
                        i_notes = cl.name2note(name = prev, stage = 4)
                        for i_note in i_notes:
                            i_note = pyd.Note(velocity = 100, pitch = i_note, 
                            start = time_shift, end = time_shift + local_duration * self.minStep)
                            chords.notes.append(i_note)
                        prev = chord
                        time_shift += local_duration * self.minStep
                        local_duration = 1
            if prev != "NC":
                i_notes = cl.name2note(name = prev, stage = 4)
                for i_note in i_notes:
                    i_note = pyd.Note(velocity = 100, pitch = i_note, 
                    start = time_shift, end = time_shift + local_duration * self.minStep)
                    chords.notes.append(i_note)
            gen_midi.instruments.append(chords)

            time_shift = 0.0
            local_duration = 0
            prev = rest_pitch
            for note in data["notes"]:
                note = int(note)
                if note < 0 or note > 129:
                    continue
                if note == hold_pitch:
                    local_duration += 1
                elif note == rest_pitch:
                    time_shift += self.minStep
                else:
                    if prev == rest_pitch:
                        prev = note
                        local_duration = 1
                    else:
                        i_note = pyd.Note(velocity = 100, pitch = prev, 
                            start = time_shift, end = time_shift + local_duration * self.minStep)
                        melodies.notes.append(i_note)
                        prev = note
                        time_shift += local_duration * self.minStep
                        local_duration = 1
            if prev != rest_pitch:
                i_note = pyd.Note(velocity = 100, pitch = prev, 
                            start = time_shift, end = time_shift + local_duration * self.minStep)
                melodies.notes.append(i_note)
            gen_midi.instruments.append(melodies)
            gen_midi.write(output)
            print("finish render midi on " + output)
        if self.datasetName == "Irish":
            # 130 one hot vectors 
            # 0-127 for pitch
            # 128 for hold 129 for rest
            rest_pitch = 129
            hold_pitch = 128
            local_duration = 0
            time_shift = 0.0
            local_duration = 0
            prev = rest_pitch
            for note in data["notes"]:
                note = int(note)
                if note < 0 or note > 129:
                    continue
                if note == hold_pitch:
                    local_duration += 1
                elif note == rest_pitch:
                    time_shift += self.minStep
                else:
                    if prev == rest_pitch:
                        prev = note
                        local_duration = 1
                    else:
                        i_note = pyd.Note(velocity = 100, pitch = prev, 
                            start = time_shift, end = time_shift + local_duration * self.minStep)
                        melodies.notes.append(i_note)
                        prev = note
                        time_shift += local_duration * self.minStep
                        local_duration = 1
            if prev != rest_pitch:
                i_note = pyd.Note(velocity = 100, pitch = prev, 
                            start = time_shift, end = time_shift + local_duration * self.minStep)
                melodies.notes.append(i_note)
            gen_midi.instruments.append(melodies)
            gen_midi.write(output)
            print("finish render midi on " + output)
            
    def text2midi(self, text_ad, recogLevel = "Mm",output = "test.mid"):
        gen_midi = pyd.PrettyMIDI()
        melodies = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
        chords = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
        if self.datasetName == "Nottingham":
            # 130 one hot vectors 
            # 0-127 for pitch
            # 128 for hold 129 for rest
            rest_pitch = 129
            hold_pitch = 128
            with open(text_ad,"r") as f:
                lines = f.readlines()
                read_flag = "none"
                for line in lines:
                    line = line.strip()
                    # if line == "Chord:":
                    #     continue
                    if line == "Chord Sequence:":
                        read_flag = "chord_seq"
                        continue
                    if line == "Notes:":
                        read_flag = "notes"
                        continue
                    if read_flag == "chord_seq":
                        cl = Chord_Loader(recogLevel = recogLevel)
                        elements = line.split(" ")
                        time_shift = 0.0
                        local_duration = 0
                        prev = "NC"
                        for chord in elements:
                            if chord == "":
                                continue
                            chord = cl.index2name(x = int(chord))
                            if chord == prev:
                                local_duration += 1
                            else:
                                if prev == "NC":
                                    prev = chord
                                    time_shift += local_duration * self.minStep
                                    local_duration = 1
                                else:
                                    i_notes = cl.name2note(name = prev, stage = 4)
                                    for i_note in i_notes:
                                        i_note = pyd.Note(velocity = 100, pitch = i_note, 
                                        start = time_shift, end = time_shift + local_duration * self.minStep)
                                        chords.notes.append(i_note)
                                    prev = chord
                                    time_shift += local_duration * self.minStep
                                    local_duration = 1
                        if prev != "NC":
                            i_notes = cl.name2note(name = prev, stage = 4)
                            for i_note in i_notes:
                                i_note = pyd.Note(velocity = 100, pitch = i_note, 
                                start = time_shift, end = time_shift + local_duration * self.minStep)
                                chords.notes.append(i_note)
                        gen_midi.instruments.append(chords)
                        continue
                    if read_flag == "notes":
                        elements = line.split(" ")
                        time_shift = 0.0
                        local_duration = 0
                        prev = rest_pitch
                        for note in elements:
                            note = int(note)
                            if note < 0 or note > 129:
                                continue
                            if note == hold_pitch:
                                local_duration += 1
                            elif note == rest_pitch:
                                time_shift += self.minStep
                            else:
                                if prev == rest_pitch:
                                    prev = note
                                    local_duration = 1
                                else:
                                    i_note = pyd.Note(velocity = 100, pitch = prev, 
                                        start = time_shift, end = time_shift + local_duration * self.minStep)
                                    melodies.notes.append(i_note)
                                    prev = note
                                    time_shift += local_duration * self.minStep
                                    local_duration = 1
                        if prev != rest_pitch:
                            i_note = pyd.Note(velocity = 100, pitch = prev, 
                                        start = time_shift, end = time_shift + local_duration * self.minStep)
                            melodies.notes.append(i_note)
                        gen_midi.instruments.append(melodies)
                        continue
                gen_midi.write(output)
                print("finish render midi on " + output)

    
    






            

