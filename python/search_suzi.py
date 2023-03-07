# search suzi of effective tile

def check_effective_hai(hai):
    if hai==0:
        return [0,1,2]
    elif hai==1:
        return [0,1,2,3]
    elif hai==2:
        return [0,1,2,3,4]
    elif hai==3:
        return [1,2,3,4,5]
    elif hai==4:
        return [2,3,4,5,6]
    elif hai==5:
        return [3,4,5,6,7]
    elif hai==6:
        return [4,5,6,7,8]
    elif hai==7:
        return [5,6,7,8]
    elif hai==8:
        return [6,7,8]
    elif hai==9:
        return [9,10,11]
    elif hai==10:
        return [9,10,11,12]
    elif hai==11:
        return [9,10,11,12,13]
    elif hai==12:
        return [10,11,12,13,14]
    elif hai==13:
        return [11,12,13,14,15]
    elif hai==14:
        return [12,13,14,15,16]
    elif hai==15:
        return [13,14,15,16,17]
    elif hai==16:
        return [14,15,16,17]
    elif hai==17:
        return [15,16,17]
    elif hai==18:
        return [18,19,20]
    elif hai==19:
        return [18,19,20,21]
    elif hai==20:
        return [18,19,20,21,22]
    elif hai==21:
        return [19,20,21,22,23]
    elif hai==22:
        return [20,21,22,23,24]
    elif hai==23:
        return [21,22,23,24,25]
    elif hai==24:
        return [22,23,24,25,26]
    elif hai==25:
        return [23,24,25,26]
    elif hai==26:
        return [24,25,26]
    else:
        return [hai]

def check_suzi(hai):
            if hai==0:
                return [3]
            elif hai==1:
                return [4]
            elif hai==2:
                return [5]
            elif hai==3:
                return [0, 6]
            elif hai==4:
                return [1,7]
            elif hai==5:
                return [2,8]
            elif hai==6:
                return [3]
            elif hai==7:
                return [4]
            elif hai==8:
                return [5]
            elif hai==9:
                return [12]
            elif hai==10:
                return [13]
            elif hai==11:
                return [14]
            elif hai==12:
                return [9,15]
            elif hai==13:
                return [10,16]
            elif hai==14:
                return [11,17]
            elif hai==15:
                return [12]
            elif hai==16:
                return [13]
            elif hai==17:
                return [14]
            elif hai==18:
                return [21]
            elif hai==19:
                return [22]
            elif hai==20:
                return [23]
            elif hai==21:
                return [18,24]
            elif hai==22:
                return [19,25]
            elif hai==23:
                return [20,26]
            elif hai==24:
                return [21]
            elif hai==25:
                return [22]
            elif hai==26:
                return [23]
            else:
                return []

def is_piece_of_syuntsu(hand,type):
            is_piece_of_syuntsu = False
            if type in [0,9,18]:
                if hand[0][type+1]==1 and hand[0][type+2]==1:
                    is_piece_of_syuntsu = True
            elif type in [1,10,19]:
                if hand[0][type-1]==1 and hand[0][type+1]==1:
                    is_piece_of_syuntsu = True
                elif hand[0][type+1]==1 and hand[0][type+2]==1:
                    is_piece_of_syuntsu = True
            elif type in [2,3,4,5,6,11,12,13,14,15,20,21,22,23,24]:
                if hand[0][type-2]==1 and hand[0][type-1]==1:
                    is_piece_of_syuntsu = True
                elif hand[0][type-1]==1 and hand[0][type+1]==1:
                    is_piece_of_syuntsu = True
                elif hand[0][type+1]==1 and hand[0][type+2]==1:
                    is_piece_of_syuntsu = True
            elif type in [7,16,25]:
                if hand[0][type-2]==1 and hand[0][type-1]==1:
                    is_piece_of_syuntsu = True
                elif hand[0][type-1]==1 and hand[0][type+1]==1:
                    is_piece_of_syuntsu = True
            elif type in [8,17,26]:
                if hand[0][type-2]==1 and hand[0][type-1]==1:
                    is_piece_of_syuntsu = True
            return is_piece_of_syuntsu