
def read_tuple_files(p):
    cool = p.copy()
    heat = p.copy()
    cool['heat'] = False
    heat['heat'] = True
    return {
        "cool": read_files_v2(cool),
        "heat": read_files_v2(heat)
    }


def substitch(heat, cool, name="S"):
    niv = sorted( list({E for E, S in heat[name].items() if not np.isinf(S)} &
                   {E for E, S in cool[name].items() if not np.isinf(S)}) )
    #chosen not infinite values
    cniv = niv[len(niv) // 3 : 2 * len(niv) // 3]
    #print(cniv)
    shift = 0
    count = 0
    for E in cniv:
        shift += heat[name][E] - cool[name][E]
        count += 1
    shift /= count

    result = od()
    for E in reversed(cool[name]):
        if E < median(cniv):
            result[E] = cool[name][E]
    for E in heat[name]:
        if E >= median(cniv):
            result[E] = heat[name][E] - shift
    return result

def stitch(heat, cool):
    #not infinite values
    return {
        "S": substitch(heat, cool, "S"),
        "magnetization": substitch(heat, cool, "magnetization"),
        "magnetization^2": substitch(heat, cool, "magnetization^2"),
        "concentration": substitch(heat, cool, "concentration"),
        "concentration^2": substitch(heat, cool, "concentration^2"),
        "L": heat["L"],
        "D": heat["D"],
        "R": heat["R"]
       }
