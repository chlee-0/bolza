import time
import numpy as np
import random
import itertools
import re
import io
#import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import pickle


def linestolist(filename):
    with io.open(filename, encoding='utf8') as thefile:
        lines = thefile.read().splitlines()
    thefile.close()        
    plist = []
    for line in lines :            
        #line = line.replace(u'\ufeff', '')
        plist.append(line)
    #plist = list(map(eval, plist))
    plist = list(plist)
    return plist


def word_files(opt=''):
    out = []
    for filename in glob.glob('bolza_'+opt+'*.txt'):
        out.append(filename)
    return out


def conj_count(length):
    out = []
    for itm in  word_files(str(length)):
        itm_re = itm.replace('bolza_'+str(length), "")
        if itm_re[0] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            out.append(itm)
    count = 0
    for itm in out:
        with io.open(itm, encoding='utf8') as thefile:
            lines = thefile.read().splitlines()
        thefile.close()        
        count+=len(lines)
    return count


def ls_files():
    out = []
    for filename in glob.glob('ls_bolza_*.txt'):
        out.append(filename)
    return out


def ls_count(gp_file, saveQ = False):
    start = time.time()
    dict_res = {}
    with io.open(gp_file, encoding='utf8') as thefile:
        lines = thefile.read().splitlines()
    thefile.close()
    count = 1
    for line in lines :
        if count%10000 ==0:
            print(count, line)
        val = toVector(tuple(map(int,line)))[:2]
        if val in dict_res:
            dict_res.update({val:dict_res[val]+1})
        else:
            dict_res.update({val:1})
        count+=1
    end = time.time()
    print(gp_file, ":", "time for processing", end-start)
    if saveQ:
        ls_file = 'ls_'+gp_file
        print('writing to {}'.format(ls_file))
        with open(ls_file, 'wb') as f:
            pickle.dump(dict_res, f)    
    return dict_res


def ls_count_parallel(opt='', saveQ = False):
    with ThreadPoolExecutor(max_workers=8) as executor:     
      tasks = []
      for itm in word_files(opt):
        task = executor.submit(lambda p:ls_count(*p), [itm, saveQ])
        tasks.append(task)
    
      for future in as_completed(tasks):
        print(future.result())


def ls_count_simple(length):
    res = {}
    for n in range(1, length+1):
        for conj_class in find_conj_classes2(n, return_list = True):
            #print(conj_class)
            len_orbit = toVector(tuple(map(int, conj_class)))[:2]
            if len_orbit in res:
                prev_val = res[len_orbit]
            else:
                prev_val = 0
            res.update({len_orbit:prev_val+1})
    return res


def ls_count_simple_nonprim(length):
    res = {}
    for conj_class in non_prim_conj_classes2(length):
        #print(conj_class)
        len_orbit = toVector(tuple(map(int, conj_class)))[:2]
        if len_orbit in res:
            prev_val = res[len_orbit]
        else:
            prev_val = 0
        res.update({len_orbit:prev_val+1})
    return res


def ls_load(filename):
    with open(filename, 'rb') as f:
        loaded_dict = pickle.load(f)    
    return loaded_dict


def ls_full_dict():
    global final_dict
    if 'final_dict' not in globals() :
        final_dict = {}
        for itm in ls_files(): 
            sign = 1
            if itm == 'ls_bolza_nonprim.txt':
                sign = -1
            temp_dict = ls_load(itm)
            for entry in temp_dict :
                val = sign*temp_dict[entry]
                if entry in final_dict:
                    final_dict.update({entry:final_dict[entry]+val})
                else:
                    final_dict.update({entry:val})
    return final_dict


def ls_check():
    tt = ls_full_dict()
    tt_nonprim = ls_load('ls_bolza_nonprim.txt')
    lines = linestolist('ls_AS_data.txt')
    for line in lines:
        val = list(map(int, line.split(',')))
        if tt[tuple(val[1:3])]!=val[0]:
            print(val, tt[tuple(val[1:3])], tt_nonprim[tuple(val[1:3])])
    for itm in ls_files():     
        temp_dict = ls_load(itm)
        for entry in temp_dict :
            if entry in [(113, 80), (85,60)]:
                print(itm, entry, temp_dict[entry]) 


def normalize(vec_list):
    x = vec_list
    pos = next((i for i, entry in enumerate(x) if entry), None)
    if x[pos] < 0:
        return tuple([-x[0], -x[1], -x[2], -x[3], -x[4], -x[5], -x[6], -x[7]])
    else:
        return vec_list


def mult_gen(vec_list, j):
    arr = np.array(vec_list, dtype=np.int32)
    if j==0:            
        mat = np.array([[1, 2, 0, 0, 1, 0, 1, 0], [1, 1, 0, 0, 0, 1, 0, 1], [0, 0, 1, 2, -1, 0, 1, 0], [0, 0, 1, 1, 0, -1, 0, 1],\
                        [1, 2, -1, -2, 1, 2, 0, 0], [1, 1, -1, -1, 1, 1, 0, 0], [1, 2, 1, 2, 0, 0, 1, 2], [1, 1, 1, 1, 0, 0, 1, 1]], dtype=np.int32)
    elif j==1:
        mat = np.array([[1, 2, 0, 0, 0, 0, 0, 2], [1, 1, 0, 0, 0, 0, 1, 0], [0, 0, 1, 2, 0, -2, 0, 0], [0, 0, 1, 1, -1, 0, 0, 0],\
                        [0, 0, -2, -2, 1, 2, 0, 0], [0, 0, -1, -2, 1, 1, 0, 0], [2, 2, 0, 0, 0, 0, 1, 2], [1, 2, 0, 0, 0, 0, 1, 1]], dtype=np.int32)
    elif j==2:
        mat = np.array([[1, 2, 0, 0, -1, 0, 1, 0], [1, 1, 0, 0, 0, -1, 0, 1], [0, 0, 1, 2, -1, 0, -1, 0], [0, 0, 1, 1, 0, -1, 0, -1],\
                        [-1, -2, -1, -2, 1, 2, 0, 0], [-1, -1, -1, -1, 1, 1, 0, 0], [1, 2, -1, -2, 0, 0, 1, 2], [1, 1, -1, -1, 0, 0, 1, 1]], dtype=np.int32)
    elif j==3:
        mat = np.array([[1, 2, 0, 0, 0, -2, 0, 0], [1, 1, 0, 0, -1, 0, 0, 0], [0, 0, 1, 2, 0, 0, 0, -2], [0, 0, 1, 1, 0, 0, -1, 0],\
                        [-2, -2, 0, 0, 1, 2, 0, 0], [-1, -2, 0, 0, 1, 1, 0, 0], [0, 0, -2, -2, 0, 0, 1, 2], [0, 0, -1, -2, 0, 0, 1, 1]], dtype=np.int32)
    elif j==4:
        mat = np.array([[1, 2, 0, 0, -1, 0, -1, 0], [1, 1, 0, 0, 0, -1, 0, -1], [0, 0, 1, 2, 1, 0, -1, 0], [0, 0, 1, 1, 0, 1, 0, -1],\
                        [-1, -2, 1, 2, 1, 2, 0, 0], [-1, -1, 1, 1, 1, 1, 0, 0], [-1, -2, -1, -2, 0, 0, 1, 2], [-1, -1, -1, -1, 0, 0, 1, 1]], dtype=np.int32)
    elif j==5:
        mat = np.array([[1, 2, 0, 0, 0, 0, 0, -2], [1, 1, 0, 0, 0, 0, -1, 0], [0, 0, 1, 2, 0, 2, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0],\
                        [0, 0, 2, 2, 1, 2, 0, 0], [0, 0, 1, 2, 1, 1, 0, 0], [-2, -2, 0, 0, 0, 0, 1, 2], [-1, -2, 0, 0, 0, 0, 1, 1]], dtype=np.int32)
    elif j==6:
        mat = np.array([[1, 2, 0, 0, 1, 0, -1, 0], [1, 1, 0, 0, 0, 1, 0, -1], [0, 0, 1, 2, 1, 0, 1, 0], [0, 0, 1, 1, 0, 1, 0, 1],\
                        [1, 2, 1, 2, 1, 2, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0], [-1, -2, 1, 2, 0, 0, 1, 2], [-1, -1, 1, 1, 0, 0, 1, 1]], dtype=np.int32)
    elif j==7:
        mat = np.array([[1, 2, 0, 0, 0, 2, 0, 0], [1, 1, 0, 0, 1, 0, 0, 0], [0, 0, 1, 2, 0, 0, 0, 2], [0, 0, 1, 1, 0, 0, 1, 0],\
                        [2, 2, 0, 0, 1, 2, 0, 0], [1, 2, 0, 0, 1, 1, 0, 0], [0, 0, 2, 2, 0, 0, 1, 2], [0, 0, 1, 2, 0, 0, 1, 1]], dtype=np.int32)
    return tuple(mat.dot(arr))


def toVector(gg):
    if gg == tuple([0]):
        return tuple([1, 1, 0, 0, 1, 1, 1, 1])
    elif gg == tuple([1]):
        return tuple([1, 1, 0, 0, 0, 0, 2, 1])
    elif gg == tuple([2]):
        return tuple([1, 1, 0, 0, -1, -1, 1, 1])
    elif gg == tuple([3]):
        return tuple([1, 1, 0, 0, -2, -1, 0, 0])
    elif gg == tuple([4]):
        return tuple([1, 1, 0, 0, -1, -1, -1, -1])
    elif gg == tuple([5]):
        return tuple([1, 1, 0, 0, 0, 0, -2, -1])
    elif gg == tuple([6]):
        return tuple([1, 1, 0, 0, 1, 1, -1, -1])
    elif gg == tuple([7]):
        return tuple([1, 1, 0, 0, 2, 1, 0, 0])
    elif gg:
        temp = mult_gen(toVector(gg[:-1]), gg[-1])
        return normalize(temp)
    else :
        return tuple([1, 0, 0, 0, 0, 0, 0, 0])


inv_pairs = {(0,4), (1, 5), (2, 6), (3, 7), (4, 0), (5, 1), (6, 2), (7, 3)}
geninv = {0:4,1:5,2:6,3:7,4:0,5:1,6:2,7:3}
enumgroup2 = {0:{''}, 1:{'0', '1', '2', '3', '4', '5', '6', '7'}}


def reduce2(ww):
    replacements_whole = [
        ("04", ""),
        ("15", ""),
        ("26", ""),
        ("37", ""),
        ("40", ""),
        ("51", ""),
        ("62", ""),
        ("73", ""),
        ("1630", "0361"),
        ("6305", "5036"),
        ("3052", "2503"),
        ("5274", "4725"),
        ("2741", "1472"),
        ("7416", "6147"),
        ("4163", "3614"),
        ("7250", "0527"),
        ("03614", "163"),
        ("14725", "274"),
        ("25036", "305"),
        ("36147", "416"),
        ("50361", "630"),
        ("61472", "741"),
        ("03613614", "163163"),
        ("14724725", "274274"),
        ("25035036", "305305"),
        ("36146147", "416416"),
        ("50360361", "630630"),
        ("61471472", "741741"),
        ("03613613614", "163163163"),
        ("14724724725", "274274274"),
        ("25035035036", "305305305"),
        ("36146146147", "416416416"),
        ("50360360361", "630630630"),
        ("61471471472", "741741741"),
        ("03613613613614", "163163163163"),
        ("14724724724725", "274274274274"),
        ("25035035035036", "305305305305"),
        ("36146146146147", "416416416416"),
        ("50360360360361", "630630630630"),
        ("61471471471472", "741741741741"),
        ("03613613613636114", "163163163163163"),
        ("14724724724724725", "274274274274274"),
        ("25035035035036503", "305305305305305"),
        ("36146146146146147", "416416416416416"),
        ("50360360360360361", "630630630630630"),
        ("61471471471471472", "741741741741741")
    ]    
    replacements = replacements_whole
    user_input = ww
    while any(pattern[0] in user_input for pattern in replacements):
        for pattern in replacements:
            user_input = user_input.replace(*pattern)
    return user_input

   
def exe_enumgroup2(length):
    if length not in enumgroup2:        
        print("working on: ", length)
        #recursive step
        start = time.time()
        if length-1 not in enumgroup2:
            exe_enumgroup2(length-1)
        end = time.time()
        print(length, ":", "time for preprocessing", end-start)        
        out = set()
        counter = 0
        for ww in enumgroup2[length - 1]:
            if counter % 10000 ==0:
                print(ww, counter)        
            for j in alphabets(1):
                user_input = ww+j
                user_input = reduce2(user_input)
                if len(user_input) == length:
                    #out.add(tuple(map(int, user_input)))
                    out.add(user_input)
            counter+=1
        enumgroup2.update({length:out})
        end = time.time()
        print(length, ":", "time of execution", end-start)
        #print(len(wordlist))
        print("number of new words", len(out))
    else:
        out = enumgroup2[length]
    return out


def cyclic_perms(a):
    n = len(a)
    #except identity
    #to include a : range(1, n)-> range(n)
    #return set([tuple([a[i - j] for i in range(n)]) for j in range(1,n)])
    return set([tuple([a[i - j] for i in range(n)]) for j in range(n)])


def cyclic_perms2(a):
    return list(map(lambda itm:''.join(itm), cyclic_perms(a)))


def cyclic_minimum2(a):
    if a == "":
        return ""
    else:
        return sorted(cyclic_perms2(a))[0]
      

def rigidQ2(ww):
    if len(ww)<2 :
        return True
    elif reduce2(ww) != ww:
        return False
    else:
        if (int(ww[0]),int(ww[-1])) not in inv_pairs:
            if ww == cyclic_minimum2(ww):
                if ww == find_conj_rep2(ww):
                    return True
                else:
                    return False
            else:
                return False


def find_conj_rep2(www):
    if www :
        perms = cyclic_perms2(www)
        word_len = len(www)
        #w_min = cyclic_minimum(www)        
        w_min = sorted(perms)[0]    
        for ww in perms:
            for j in alphabets(1):
                newword = j + ww + str(geninv[int(j)])            
                conjugate_word = reduce2(newword)
                if len(conjugate_word) < word_len:
                    c_min = cyclic_minimum2(conjugate_word)
                    #return c_min
                    return find_conj_rep2(c_min)
                elif len(conjugate_word) == word_len and conjugate_word not in perms:                    
                    c_min = cyclic_minimum2(conjugate_word)
                    if c_min<=w_min:
                        #print(ww, newword, conjugate_word)
                        return find_conj_rep2(c_min)
                        #return c_min
        return w_min
    else:
        return ''


def find_conj_classes2(length, return_list = False, write_file = False, starting_letter = ''):
    print("working on: ", length, starting_letter)
    start = time.time()
    counter = 0
    out = []
    ans = 0
    if write_file:
        fname = 'bolza_'+str(length)
        if starting_letter:
            fname = fname + "_"+starting_letter
        fname = fname + '.txt'
        thefile = io.open(fname, 'a', encoding='utf8')
    if starting_letter:
        for x in itertools.product(alphabets(1), repeat=length-len(starting_letter)):
            ww = starting_letter+''.join(x)
            if rigidQ2(ww):
                ans+=1
                if return_list:
                    out.append(ww)
                if write_file:
                    thefile.write("%s\n" % ww)
            counter+=1                    
    else:
        for x in itertools.product(alphabets(1), repeat=length):
            ww = ''.join(x)        
            if rigidQ2(ww):            
                ans+=1
                if return_list:
                    out.append(ww)
                if write_file:
                    thefile.write("%s\n" % ww)            
            # if counter % 100000 ==0:
            #     print(ww, counter)
            counter+=1
    end = time.time()    
    print(length, ":", "time of execution", end-start)
    if write_file:
        thefile.close()      
    if return_list:
        return out
    else:
        return ans


def alphabets(length):
    base = ["0", "1","2","3","4","5","6","7"]
    # if length == 1:
    #     return base
    # else :
    out = []
    for x in itertools.product(base, repeat=length):
        ww = ''.join(x)
        if ww == cyclic_minimum2(ww):
            if ww == reduce2(ww):        
                out.append(ww)
    return out


def find_conj_classes2_parallel(length, prefix_list):
    #prefix_list = ["0", "1","2","3","4","5","6","7"]
    random.shuffle(prefix_list)
    with ThreadPoolExecutor(max_workers=8) as executor:     
      tasks = []
      for prefix in prefix_list:
        task = executor.submit(lambda p: find_conj_classes2(*p), [length, False, True, prefix])      
        tasks.append(task)
    
      for future in as_completed(tasks):
        print(future.result())


def non_prim_conj_classes2(length, write_file = False):
    out = set() 
    res = []
    if write_file:
        thefile = io.open('bolza_nonprim.txt', 'a', encoding='utf8')
    for k in [1,2,3,4,5,6]:
        exe_enumgroup2(k)
        for itm in enumgroup2[k]:
            word = itm
            for n in [1,2,3,4,5,6,7,8,9,10,12]:
                word += itm
                ww = reduce2(word)
                if len(ww) <= length:
                    if rigidQ2(ww):
                        if ww not in out:
                            out.add(ww)
                            res.append(ww)
                            if write_file:
                                thefile.write("%s\n" % ww)
                        if len(word) < len(ww):
                            print(word, ww)
    return res