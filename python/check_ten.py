# 他家と自分の点数の差を返す feat[i]: i+1位との差
def check_ten_difference(tens,my_rank) -> list:
    sorted_tens = sorted(tens,reverse=True)
    my_score = sorted_tens[my_rank-1]
    for i in range(4):
        sorted_tens[i] -= my_score
    return sorted_tens

# 指定したプレイヤーとの点差を返す
def diff_ten(tens,my_rank,target_rank) -> int:
    diff_ten_list = check_ten_difference(tens,my_rank)
    return diff_ten_list[target_rank-1]

# 着順アップできるかどうか判定
def is_possible_to_rank_up(tens,my_rank,dora_num,dealer_num,kyotaku,honba) -> bool:
    if my_rank==1:
        return True
    diff_ten_list = check_ten_difference(tens,my_rank)
    necessary_ten = diff_ten_list[my_rank-2]
    expected_ten = 1000
    if dora_num==1:
        expected_ten = 2000
    elif dora_num==2:
        expected_ten = 4000
    elif dora_num==3:
        expected_ten = 7800
    elif dora_num==4:
        expected_ten = 8000
    elif dora_num==5:
        expected_ten = 12000
    elif dora_num==6:
        expected_ten = 12000
    elif dora_num==7:
        expected_ten = 16000
    elif dora_num==8:
        expected_ten = 16000
    elif dora_num==9:
        expected_ten = 24000
    
    if dealer_num==0:
        expected_ten *= 1.5

    kyotaku_num = 0
    honba_num = 0
    for i in range(5):
        if kyotaku[i][0]==1:
            kyotaku_num += 1
        if honba[i][0]==1:
            honba_num += 1
    
    expected_ten += kyotaku_num*1000+honba_num*300
    if expected_ten>=necessary_ten:
        return True
    else:
        return False
