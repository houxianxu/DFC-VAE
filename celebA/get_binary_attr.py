import pandas as pd

def get_attr_list(df, attr):
    positive = df['index'][df[attr] == 1]
    negative = df['index'][df[attr] == -1]
    pos_list = positive.tolist()
    pos_str = attr + '_pos' + ',' + ','.join(pos_list) + '\n'
    neg_list = negative.tolist()
    neg_str = attr + '_neg' + ',' + ','.join(neg_list) + '\n'
    return pos_str + neg_str

def main():
    attr_file = 'list_attr_celeba.txt'
    attr_extracted = '5_o_Clock_Shadow,Arched_Eyebrows,Attractive,Bags_Under_Eyes,Bald,Bangs,Big_Lips,Big_Nose,Black_Hair,Blond_Hair,Blurry,Brown_Hair,Bushy_Eyebrows,Chubby,Double_Chin,Eyeglasses,Goatee,Gray_Hair,Heavy_Makeup,High_Cheekbones,Male,Mouth_Slightly_Open,Mustache,Narrow_Eyes,No_Beard,Oval_Face,Pale_Skin,Pointy_Nose,Receding_Hairline,Rosy_Cheeks,Sideburns,Smiling,Straight_Hair,Wavy_Hair,Wearing_Earrings,Wearing_Hat,Wearing_Lipstick,Wearing_Necklace,Wearing_Necktie,Young'   # attr_extracted = 'Smiling'
    df = pd.read_csv(attr_file, delim_whitespace=True)

    i = 0
    with open('attr_binary_list.txt', 'w') as f:
        for attr in attr_extracted.split(','):
            attr_str = get_attr_list(df, attr)
            print(len(attr_str))
            f.write(attr_str)

if __name__ == '__main__':
    main()


