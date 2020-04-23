def parse_mtfmapper_sfr_data(filename, pixelpitch):
    with open(filename, 'r') as f:
        lines = f.readlines()
        data = []
        demo_data = lines[0].split()
        if float(demo_data[5]) < 10:
            left_tan = True
        for pair in zip(lines[::2], lines[1::2]):
            # the left item in the tuple is the odd numbered row, the right item in the tuple is the even numbered row
            contents1 = [float(i) for i in pair[0].split()]
            contents2 = [float(i) for i in pair[1].split()]
            loc_x = (contents1[1] + contents2[1]) / 2
            loc_y = (contents1[2] + contents2[2]) / 2
            if left_tan:
                mtf_tan = contents1[5:]
                mtf_sag = contents2[5:]
            else:
                mtf_tan = contents2[5:]
                mtf_sag = contents1[5:]

            data.append({
                'pixel_x': loc_x,
                'pixel_y': loc_y,
                'mtf_tan': mtf_tan,
                'mtf_sag': mtf_sag,
                })

    return dict(
        mtf_unit = [x/64/(pixelpitch/1000) for x in range(0,64)], # mtfmapper yields normalized frequency in k/64 cy/px for k=(0..63)
        data = data
        )