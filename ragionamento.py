from pyswip import Prolog
import csv

colonne = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep',
           'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width',
           'ram', 'sc_h', 'sc_w', 'talk_time']


def creaKnowledgeBase(dataset):
    # Creazione file prolog
    output_file = 'knowledge_base.pl'
    with open(output_file, 'w') as f:
        f.write('%---Knowledge Base---\n\n')
        f.write(f'% telefono({", ".join(dataset.columns)})\n\n')

        for index, row in dataset.iterrows():
            valori = ", ".join([str(int(v)) if isinstance(v, float) and v.is_integer() else str(v) for v in row.values])
            f.write(f"telefono({valori}).\n")

    print(f"Conversione completata! Creato il file: {output_file}")
    return output_file


def scrivi_regole(output_file, dataset):
    with open(output_file, 'a') as f:
        f.write("\n\n")

        for col in colonne:
            for fascia in range(4):
                subset = dataset[dataset['price_range'] == fascia]
                if col == 'clock_speed' or col == 'm_dep':
                    min_val = float(subset[col].min())
                    max_val = float(subset[col].max())
                else:
                    min_val = int(subset[col].min())
                    max_val = int(subset[col].max())
                f.write(f"range_{col}({fascia}, {min_val}, {max_val}).\n")
            f.write("\n")

    with open(output_file, 'a') as f:
        f.write(":- use_module(library(random)).\n")
        f.write("genera_intero(Min, Max, X) :- random(Min, Max, X).\n")
        f.write("genera_decimale(Min, Max, Out) :- random(Min, Max, X), Out is round(X * 10) / 10.0.\n")
        f.write("vincolo_connettivita(FourG, ThreeG) :- (FourG =:= 1 -> ThreeG = 1 ; random(0 , 2, ThreeG)).\n")

        f.write(
            "genera_nuovo_telefono(PR, telefono(Batt, Blue, Clock, DS, FC, FourG, Mem, Dep, Wt, Cores, PC, PxH, PxW, Ram, ScH, ScW, Talk, ThreeG, Touch, Wifi, PR)) :-\n")

        f.write("range_battery_power(PR, MinBat, MaxBat), genera_intero(MinBat, MaxBat, Batt),\n")
        f.write("range_clock_speed(PR, MinClk, MaxClk), genera_decimale(MinClk, MaxClk, Clock),\n")
        f.write("range_fc(PR, MinFC, MaxFC), genera_intero(MinFC, MaxFC, FC),\n")
        f.write("range_int_memory(PR, MinMem, MaxMem), genera_intero(MinMem, MaxMem, Mem),\n")
        f.write("range_m_dep(PR, MinDep, MaxDep), genera_decimale(MinDep, MaxDep, Dep),\n")
        f.write("range_mobile_wt(PR, MinWt, MaxWt), genera_intero(MinWt, MaxWt, Wt),\n")
        f.write("range_n_cores(PR, MinCores, MaxCores), genera_intero(MinCores, MaxCores, Cores),\n")
        f.write("range_pc(PR, MinPC, MaxPC), genera_intero(MinPC, MaxPC, PC),\n")
        f.write("range_px_height(PR, MinPxH, MaxPxH), genera_intero(MinPxH, MaxPxH, PxH),\n")
        f.write("range_px_width(PR, MinPxW, MaxPxW), genera_intero(MinPxW, MaxPxW, PxW),\n")
        f.write("range_ram(PR, MinRam, MaxRam), genera_intero(MinRam, MaxRam, Ram),\n")
        f.write("range_sc_h(PR, MinScH, MaxScH), genera_intero(MinScH, MaxScH, ScH),\n")
        f.write("range_sc_w(PR, MinScW, MaxScW), genera_intero(MinScW, MaxScW, ScW),\n")
        f.write("range_talk_time(PR, MinTalk, MaxTalk), genera_intero(MinTalk, MaxTalk, Talk),\n")

        f.write("random(0, 2, Blue),\n")
        f.write("random(0, 2, DS),\n")
        f.write("random(0, 2, FourG),\n")
        f.write("vincolo_connettivita(FourG, ThreeG),\n")
        f.write("random(0, 2, Touch),\n")
        f.write("random(0, 2, Wifi).\n")



def inventa_telefono(fascia):
    prolog = Prolog()
    prolog.consult('knowledge_base.pl')
    #print("Inventando telefono di fascia: ", fascia)
    query = f"genera_nuovo_telefono({fascia}, Telefono)"

    # Prendiamo un risultato a caso tra quelli possibili
    risultati = list(prolog.query(query))
    import random
    if risultati:
        scelta = random.choice(risultati)
        return scelta
    return None


def crea_nuovo_dataset():
    prolog = Prolog()
    prolog.consult('knowledge_base.pl')

    with open("dataset_sintetico.csv", "w", newline='') as f:
        writer = csv.writer(f)
        header = 'battery_power,blue,clock_speed,dual_sim,fc,four_g,int_memory,m_dep,mobile_wt,n_cores,pc,px_height,px_width,ram,sc_h,sc_w,talk_time,three_g,touch_screen,wifi,price_range'
        v = header.split(',')
        writer.writerow(v)
        for i in range(1, 500):
            for j in range(0, 4):
                #Creo nuovo telefono di fascia j
                t = inventa_telefono(j)
                #Lo trasformo in stringa per poterlo scrivere sul file csv
                string_t = str(t)

                #Prendo solo i valori tra parentesi per porterli scrivere sul file csv
                inizio = string_t.find('(') + 1
                fine = string_t.find(')')
                valori_telefono = string_t[inizio:fine]

                lista_valori = valori_telefono.split(', ')
                riga_csv = []
                for v in lista_valori:
                    num = float(v)
                    if num.is_integer():
                        riga_csv.append(int(num))
                    else:
                        riga_csv.append(num)

                writer.writerow(riga_csv)