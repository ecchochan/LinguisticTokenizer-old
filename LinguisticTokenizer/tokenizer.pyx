
#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#distutils: language = c++

from __future__ import print_function


from libcpp.utility cimport pair

'''
python3.6 cythonize.py build_ext --inplace && python3 cythonize.py build_ext --inplace
'''
try:
    import __builtin__
except ImportError:
    # Python 3
    import builtins as __builtin__

from time import sleep
def print(*args, **kwargs):
    #sleep(0.04)
    return __builtin__.print(*args, **kwargs)

import pandas as pd
#from weighted_levenshtein import lev as lev_dist

import numpy as np
cimport numpy as np
from numpy cimport ndarray
cdef extern from "stdlib.h":
    ctypedef unsigned int size_t
    size_t strlen(char *s)
    void *malloc(size_t size)
    void free(void *ptr)
    int strcmp(char *a, char *b)


cdef inline size_t imin(int a, int b, int c, bint no_a):
    if a < b:
        if c < a:
            return c
        if no_a:
            return a - 1
        return a
    if c < b:
        return c
    return b


cdef int levenshtein(char *a, 
                     char *b,
                     ndarray[np.uint8_t, ndim=1, cast=True] zero_penalties ):
    cdef int m = strlen(a), n = strlen(b)
    cdef char *ctmp
    cdef int i = 0, j = 0, retval
    cdef int achr, bchr
     
    if strcmp(a, b) == 0:
        return 0

    if m > n:
        ctmp = a;
        a = b;
        b = ctmp;
        #a, b = b, a
        m, n = n, m
        

    cdef char *m1 = <char *>malloc((n + 2) * sizeof(char))
    cdef char *m2 = <char *>malloc((n + 2) * sizeof(char))
    
    for i from 0 <= i <= n:
        m1[i] = i
        m2[i] = 0

    for i from 0 <= i <= m:
        m2[0] = i + 1
        achr = a[i]
        for j from 0 <= j <= n:
            bchr = b[j]
            if achr == bchr:
                m2[j + 1] = m1[j]
            else:
                m2[j + 1] = 1 + imin(m2[j], m1[j], m1[j + 1], zero_penalties[j])

        m1, m2 = m2, m1

    retval = m1[n + 1]
    free(m2)
    free(m1)
    return retval




#from Levenshtein import distance as lev_dist

import json
import re
from itertools import islice

from collections import defaultdict

import os
import sys


#directory = '/'.join(sys.modules[__name__].__file__.split('/')[:-1])
directory = ''
from pkg_resources import resource_filename


package_name = 'LinguisticTokenizer'
import os.path

def get_file(x):
    return resource_filename(package_name, 'resources/' + x)

    
if not os.path.isfile(get_file('words2.txt')):
    raise Exception('Resource files not found.')

less_than_char_length_3_max_count = 4
less_than_char_length_2_max_count = 6


indices_pairs = [
    
    sorted([(i,j) for i in range(length) for j in range(length) if i != j and j > i],key=lambda x:x[1]-x[0],reverse=True)
    for length in range(20)

]
from libcpp.vector cimport vector
cdef vector[vector[int]] indices_pairs_i = [[i for i, j in indices] for indices in indices_pairs]
cdef vector[vector[int]] indices_pairs_j = [[j for i, j in indices] for indices in indices_pairs]

from importlib import reload

try:
    import rules
except:
    from . import rules

def argsort(list seq, indexes=None):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)) if indexes is None else indexes, key=seq.__getitem__)

def argsort(list seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

cdef class LinguisticTokenizer():
    cdef public :
        dict prefixes, suffixes, vocabs, vocab2idx, idx2root, expand_allowed, expand_allowed_reverse, pairs_exceptions, expand_exceptions, expand_exceptions_reverse, very_exceptions, cache
        set root_only, all_words, single_words_can_be_en_word, single_words
        int vocab_index
        

        
    def handle_Xfixes(self, str p, str s, str m, new_members):
        
        prefixes = self.prefixes
        suffixes = self.suffixes
        
        if p:
            first = True
            for p in p.split(','):
                if '(' in p:
                    _p = [re.sub('\(.*?\)','',p), p.replace('(','').replace(')','')]
                else:
                    _p = [p]
                for p in _p:
                    p = p.strip()
                    if p:
                        if p.startswith('#'):
                            continue
                        if p.startswith('-'):
                            p = p[1:]
                            if not p:
                                continue
                                
                            new_members.add(p)
                            if p not in suffixes:
                                suffixes[p] = set()


                            if s and first:

                                for s in s.split(','):
                                    s = s.strip()
                                    if s:
                                        #minus = s.startswith('-')
                                        if s.startswith('+'): # or minus:
                                            s = s[1:]
                                        else:
                                            raise
                                        suffixes[p].add(s)
                                        #suffixes[p].add(('e', False))
                                        #if len(s) > 1: # and not minus:
                                        #    suffixes[p].add((s[0], False))
                            if not m:
                                suffixes[p].add('') #('', False))
                                # suffixes[x] can be a list with first element equals ''
                                
                        elif p.endswith('-'):
                            p = p[:len(p)-1]
                            if not p:
                                continue
                            new_members.add(p)
                            if p not in prefixes:
                                prefixes[p] = set()
                            if not m:
                                prefixes[p].add('')
                            if s and first:
                                for s in s.split(','):
                                    s = s.strip()
                                    if s:
                                        prefixes[p].add(s)
                                        
                            # add o
                            if len(p) > 3 and not p[len(p)-1] not in 'oa':
                                new_members.add(p+'o')
                                prefixes[p+'o'] = prefixes[p]
                    first = False

            
            
        
    cdef bint check_seen(self,
                          str orig,
                          tuple bucket1,
                          str word,
                          tuple bucket2,
                          tuple fix_bucket1,
                          tuple fix_bucket2,
                          unsigned int pos,
                          set seen):
        cdef unsigned int i, L, orig_len = len(orig), less_than_char_length_2, less_than_char_length_3
        key = bucket1 + ((word,) if word else ()) + bucket2
        if fix_bucket1 is not None:
            key = fix_bucket1 + key
        if fix_bucket2 is not None:
            key = key + fix_bucket2
            
        less_than_char_length_2 = 0
        less_than_char_length_3 = 0
        for i in range(len(key)):
            L = len(key[i])
            if L < 2:
                less_than_char_length_2 += 1
            elif L < 3:
                less_than_char_length_3 += 1
                
        #less_than_char_length_2 = sum(1 for e in key if len(e) < 2)
        
        #if orig_len > less_than_char_length_2_max_count:
        #    if less_than_char_length_2 > less_than_char_length_2_max_count:
        #        return True
        
        #less_than_char_length_3 = sum(1 for e in key if len(e) < 3)
        #if orig_len > less_than_char_length_3_max_count:
        #    if less_than_char_length_3 > less_than_char_length_3_max_count:
        #        return True
        
        
        key = key + (pos,)
        if key in seen:
            return True
        seen.add(key)
        return False
                    
    cpdef combine(self, str orig, tuple pieces, unsigned int pos, bint debug):
        cdef unsigned int g, i, j, jj, index = 0, max_index = len(pieces), size
        cdef vector[int] indices_i, indices_j
        cdef tuple new_pieces
        cdef str trial, next_one, c, t
        cdef bint temp_bool, suf, pre
        cdef bint (*check_violate)(LinguisticTokenizer, tuple, unsigned int, bint)
        check_violate = self.check_violate
        all_words = self.all_words
        
        
        
        #combine_exceptions = self.combine_exceptions
        suffixes = self.suffixes
        prefixes = self.prefixes
        #for j in range(len(pieces) - 1):
        #    k = j + 1
        #    if pieces[k] in ['er','est','ing','able','ive','ative','ation','ion','es','s','']
        #for e in pieces:
        #    if e+'e' in tokenizer.all_words and e[-1]:
        
        new_pieces = pieces
        
        #for i, j in indices_pairs[]:
        indices_i = indices_pairs_i[max_index]
        indices_j = indices_pairs_j[max_index]
        size = indices_i.size()
        while index < size:
            i = indices_i[index]
            j = indices_j[index]
            index += 1
            if debug:
                print((i, j))

            if check_violate(self, new_pieces, pos, False):
                if debug:
                    print("violate")
                break
            
            
            trial = ''.join(pieces[i:j+1])
            #if trial_joined in combine_exceptions:
            #    if debug:
            #        print('bye 1')
            #    continue
            length = j-i+1
            g = 0
            
            #for t in trial:
            for jj in range(i,j+1):
                t = pieces[jj]
                if t in suffixes:
                    if '' not in suffixes[t]:
                        g = 9999
                        if debug:
                            print('bye 2')
                        break
                        
                    g += 1
                    
            if g > length:
                if debug:
                    print('bye 2.5')
                continue
            elif g == length:
                if trial in all_words:
                    if debug:
                        print('bye 3')
                    continue
                if trial in suffixes and '' in suffixes[trial]:
                    if debug:
                        print('bye 4')
                    continue
                
                
                
            #trial = trial_joined
            #suf = (trial in suffixes and empty_suffix in suffixes[trial] )
            #pre = trial in prefixes
            #root = trial in all_words
            if (trial in all_words or \
                trial in prefixes or \
                (trial in suffixes and '' in suffixes[trial] )):# and trial in orig:
                
                if j+1 < len(pieces):
                    next_one = pieces[j+1]
                    suf = next_one in suffixes
                    
                    #if suf and not(pre or root):
                    if suf:
                        pre = next_one in prefixes
                        if not pre:
                            
                            root = next_one in all_words

                            if not root:
                                #if not any(trial.endswith(c) for c, minus in suffixes[next_one] if not minus):
                                temp_bool = False
                                
                                for c in suffixes[next_one]:
                                    if trial.endswith(c):
                                        temp_bool = True
                                        break
                                        
                                if not temp_bool:
                                    if debug:
                                        print('bye 5')
                                    continue

                    
                    
                if debug:
                    print('yes', pieces[:i] + (trial,) + pieces[j+1:])
                new_pieces = pieces[:i] + (trial,) + pieces[j+1:]
                if pos > i and pos <= j:
                    pos = i
                elif pos > j:
                    pos -= j - i


        return new_pieces, pos
    
    def _check_violate(self, tuple pieces, unsigned int pos, bint debug):
        return self.check_violate( pieces, pos, debug)
    
    cdef bint check_violate(self, tuple pieces, unsigned int pos, bint debug):
        cdef unsigned i, j, jj, len_pieces = len(pieces), single_word_exist_count = 0, single_words_can_be_en_word_count = 0, okay
        cdef bint root, is_root, pre, suf, before_last
        
        pairs_exceptions = self.pairs_exceptions

        expand_allowed = self.expand_allowed
        expand_allowed_reverse = self.expand_allowed_reverse
        single_words = self.single_words
        single_words_can_be_en_word = self.single_words_can_be_en_word
        
        expand_exceptions = self.expand_exceptions
        
        prefixes = self.prefixes
        suffixes = self.suffixes
        all_words = self.all_words
        if len_pieces == 0:
            return False
        
        if len_pieces > 0:
            p = pieces[0]
            pre = p in prefixes
            root = p in all_words
            suf = p in suffixes
            if suf and not root and not pre:
                if debug:
                    print(1)
                return True
                
            
            p = pieces[len_pieces-1]
            pre = p in prefixes
            root = p in all_words
            suf = p in suffixes
            if pre and not root and not suf:
                if debug:
                    print(1)
                return True
                
            if len_pieces == 1 and not (root or pre or suf):
                if debug:
                    print(2)
                return True
                

        jj = 0
        last = None
        
        for jj in range(len_pieces):
        #for e in pieces:
            e = pieces[jj]
            is_root = jj == pos
            key = e
            '''
            if not is_root:
                #if e in suffix_last and i != len_pieces - 1:
                #    if debug:
                #        print(3)
                #    return True
                
                if last is not None and e in suffixes and last in suffixes[e]:
                    if debug:
                        print(3.5)
                    return True
                '''
                
            last = e

            if e in single_words:
                single_word_exist_count += 1
            if e in single_words_can_be_en_word:
                single_words_can_be_en_word_count += 1
                
                
            if jj+1 < len_pieces:
                w = pieces[jj+1]
                if w in expand_allowed_reverse:
                    allowed_reverse = expand_allowed_reverse[w]
                    if allowed_reverse is None or e not in allowed_reverse:
                        if debug:
                            print(4.5)
                        return True
                if is_root and e in expand_allowed:
                    allowed = expand_allowed[e]
                    if allowed is None or w not in allowed:
                        if debug:
                            print(4)
                        return True



                    '''
                    for a in expand_allowed[e]:
                        if len(pieces) > 1 if len(a) == 0 else all(i+j < len(pieces) and pieces[i+j] != v for j, v in enumerate(a)):
                            #print('!', all(i+j < len(pieces) and pieces[i+j] != v for j, v in enumerate(a)))
                            if debug:
                                print(4)
                            return True
    '''

                if e in expand_exceptions:
                    t = expand_exceptions[e][int(is_root)]  # [['ation'],['ate']]

                    if t is not None:
                        if pieces[jj+1] in t or len(t) == 0:
                            if debug:
                                print(5)
                            return True


                        

            #jj += 1
            
        if single_word_exist_count != 0 and single_word_exist_count != len_pieces and single_word_exist_count != single_words_can_be_en_word_count:
            if debug:
                print(5.5)
            return True
            
            
        for i in range(len_pieces - 1):
            a = pieces[i]
              

            #for k, v in expand_exceptions_char:
            #    if a.endswith(k) and b.startswith(v):
            #        if debug:
            #            print(6)
            #        return True
            #if a in expand_exceptions_char_left_word and any(b.startswith(c) for c in expand_exceptions_char_left_word[a]):
            #    if debug:
            #        print(7)
            #    return True

            if a in pairs_exceptions:
                b = pieces[i+1] #''.join(pieces[i+1:])
                if b in pairs_exceptions[a] :
                    if debug:
                        print(8)
                    return True


        return False
    def __init__(self):
        pass
        
            
        
    def init(self,
             enable_single_words=True):
        cdef str w,p,s
        import numpy as np
        reload( rules )
        #insert_costs = self.insert_costs = np.ones(128, dtype=np.float64) 
        #substitute_costs = self.substitute_costs = np.ones((128, 128), dtype=np.float64) * 2
        #self.cache_lev = cache_lev = {}
        '''
        def lev(x, y):
            with self.cache_lock_lev:
                if x not in cache_lev:
                    cache_lev[x] = c = {}
                else:
                    c = cache_lev[x]
                if y in c:
                    return c[y]
                else:
                    c[y] = ret = levenshtein(x,y)
                    return ret
            '''
            
        
        #self.lev = lev

        #fn = get_file('words.csv')
        #self.words = words = pd.read_csv(fn, sep=',',header=None, encoding = "ISO-8859-1")
        #self.words = words = words.where((pd.notnull(words)), None).values
        
        fn = get_file('words2.txt')
        with open(fn) as f:
            words = f.read()
        
        self.all_words = all_words = set()


        self.prefixes = prefixes = {}
        self.suffixes = suffixes = {}
        self.root_only = root_only = set()
        self.vocabs = vocabs = {}
        self.vocab2idx = vocab2idx = {}
        self.idx2root = idx2root = {}
        self.vocab_index = 0
        
        handle_Xfixes = self.handle_Xfixes
        

        '''
        for e in words:
            w = e[0]
            p = e[1]
            s = e[2]
            m = e[3]
            '''
        for line in words.split('\n'):
            if '|' in line:
                splitted = line.split('|')
                p = splitted[0].lower().strip()
                s = splitted[1].lower().strip() if len(splitted) > 1 else ''
                m = splitted[2].lower().strip() if len(splitted) > 2 else ''
                new_members = set()
                handle_Xfixes(p,s, m, new_members)
            
            else:
                w = line.lower().strip()
                new_members = set([e.strip() for e in line.lower().strip().split(',') if e.strip() and not e.startswith('#')])
                
                for w in list(new_members):
                    if w.startswith('-') or w.endswith('-'):
                        handle_Xfixes(w,None, None, new_members)
                    else:
                        if w not in vocabs:
                            idx = self.vocab_index
                            vocabs[w] = set([w])
                            vocab2idx[w] = idx
                            if idx not in idx2root:
                                idx2root[idx] = w
                            self.vocab_index += 1


                        all_words.add(w)


            synonyms = None
            for n in new_members:
                if n in vocabs:
                    synonyms = vocabs[n]
                    idx = vocab2idx[n]
                    break

            if synonyms is None:
                idx = self.vocab_index
                self.vocab_index += 1
                synonyms = set()


            for n in new_members:
                synonyms.add(n)
                vocabs[n] = synonyms
                vocab2idx[n] = idx
                if idx not in idx2root:
                    idx2root[idx] = n

                        
                        
                        
                        
                        
                        
                        
                        
        self.pairs_exceptions = rules.pairs_exceptions
        self.expand_allowed = rules.expand_allowed
        self.expand_allowed_reverse = rules.expand_allowed_reverse
        self.single_words = set()
        self.single_words_can_be_en_word = set()

        if enable_single_words:
            with open(get_file('single_words.txt'), 'r')  as f:
                single_words = [ e.strip() for e in f.read().split('\n') if e]

            for e in single_words:
                L = len(e)
                if L <= 1:
                    continue
                
                #self.expand_allowed[e] = set(['s','y','ry','son','ward','en', 'o','a','e','i','u','n','d']+ single_words)
                if e.startswith('#'):
                    continue
                if e.startswith('*'):
                    e = e[1:]
                    self.single_words_can_be_en_word.add(e)
                elif L == 2:
                    self.single_words_can_be_en_word.add(e)
                    
                self.single_words.add(e)

                w = e
                if w not in vocabs:
                    idx = self.vocab_index
                    vocabs[w] = set([w])
                    vocab2idx[w] = idx
                    if idx not in idx2root:
                        idx2root[idx] = w
                    self.vocab_index += 1
                if L > 2:
                    all_words.add(w)

        #self.verb_only = rules.verb_only
        #self.noun_only = rules.noun_only
        single_words = self.single_words
        
        #self.expand_exceptions_char = rules.expand_exceptions_char
        
        #self.expand_exceptions_char_left_word = rules.expand_exceptions_char_left_word
        
        #self.suffix_last = rules.suffix_last
        expand_exceptions = rules.expand_exceptions
        for e in vocabs:
            if e.endswith('e'):
                for m in (True, False):
                    key = (e,m) 
                    if key not in expand_exceptions:
                        expand_exceptions[key] = ''
                    expand_exceptions[key] += ',ry'
                        
        
        #self.combine_exceptions = rules.combine_exceptions
        '''
        expand_exceptions_char = rules.expand_exceptions_char
        
        for e in all_words:
            for a, b in expand_exceptions_char:
                if e.endswith(a):
                    targets = []
                    for u in all_words:
                        if u.startswith(b):
                            targets.append([u])
                            
                    for m in (True,False):
                        key = (e,m)
                        if key not in expand_exceptions:
                            expand_exceptions[key] = []
                        expt = expand_exceptions[key]
                        for t in targets:
                            if any(len(u) == 1 and u[0] == t[0] for u in expt):
                                continue
                            expt.append(t)
        '''
        
        for e in all_words:
            if e not in prefixes and e not in suffixes:
                root_only.add(e)
                            
        _expand_exceptions = {}
        self.expand_exceptions_reverse = expand_exceptions_reverse = {}
        for (a,b), v in expand_exceptions.items():
            v = set([e for e in v.split(',') if e])
            b = int(b)
            for e in v:
                if not e:
                    continue
                if e not in expand_exceptions_reverse:
                    expand_exceptions_reverse[e] = [None, None]

                expand_exceptions_reverse[e][b] = temp = set()
                temp.add(a)
            if a not in _expand_exceptions:
                _expand_exceptions[a] = expt = [None,None]
            else:
                expt = _expand_exceptions[a]
                
            if expt[b] is not None:
                raise repr((a,b)) + ' @ expand_exceptions'
                
            expt[b] = v
                            
        self.expand_exceptions = _expand_exceptions
        self.very_exceptions = very_exceptions = {}
        with open(get_file('irreguler_exceptions.json'), 'r')  as f:
            j = json.load(f)

        j = {k:v for k, v in j.items() if k != v[0]}

        for k, v in j.items():
            if not k:
                continue
            w = k
            '''
            if w not in vocabs:
                idx = self.vocab_index
                vocabs[w] = set([w])
                vocab2idx[w] = idx
                if idx not in idx2root:
                    idx2root[idx] = w
                all_words.add(w)
                self.vocab_index += 1'''
            very_exceptions[k] = tuple(v)

        from multiprocessing import Lock
        #self.cache_lock = cache_lock = Lock()
        #self.cache_lock_lev = cache_lock_lev = Lock()
        self.cache = {}
        

        
    cdef int num_roots(self, 
                        tuple x, 
                        unsigned int pos,
                        bint debug):
        cdef unsigned int  __PREFIX = 2
        cdef unsigned int __ROOT   = 3
        cdef unsigned int __SUFFIX = 4
        cdef unsigned int i
        cdef bint last_can_be_prefix = False
        cdef bint last_can_be_root   = False
        cdef bint last_can_be_suffix = False
        cdef int count = 0

        cdef bint can_be_prefix, can_be_root, can_be_suffix, not_okay, have_root
        cdef set all_words = self.all_words
        cdef dict prefixes = self.prefixes
        cdef dict suffixes = self.suffixes
        '''
        __PREFIX = 2
        __ROOT   = 3
        __SUFFIX = 4
        last_can_be_prefix = False
        last_can_be_root   = False
        last_can_be_suffix = False
        count = 0
        first = True
        '''
        have_root = False
        i = 0
        for e in x:
            
            can_be_prefix = e in prefixes and i != pos
            can_be_root   = (e in all_words and len(e) > 2) or i == pos
            can_be_suffix = e in suffixes and i != pos

            if debug:
                print(e,':', last_can_be_prefix, last_can_be_root, last_can_be_suffix)
                print(len(e)*' ',':', can_be_prefix, can_be_root, can_be_suffix)

            not_okay = True

            if can_be_suffix:
                if last_can_be_prefix or last_can_be_root or last_can_be_suffix:
                    #can_be_prefix = False
                    #can_be_root = False
                    not_okay = False
            if can_be_root:
                have_root = True
                if last_can_be_prefix:
                    #can_be_prefix = False
                    not_okay = False
            if can_be_prefix:
                if last_can_be_prefix:
                    not_okay = False


            if not last_can_be_prefix and not_okay:
                if debug:
                    print(len(e)*' ',':', '+1')
                count += 1
            last_can_be_prefix = can_be_prefix
            last_can_be_root   = can_be_root
            last_can_be_suffix = can_be_suffix
            
            i += 1

        if not have_root:
            count += 1
        return count
        
    def tokenize(self, 
               word,  
               master_bucket=None, 
               tuple fix_bucket1=None, 
               tuple fix_bucket2=None, 
               #int morphed=0, 
               bint return_all=False, 
               unsigned int max_N = 7,
               bint debug=False,
               bint debug_slow=False,
               bint short=False,
               int max_length = 30,
               bint use_cache=False,
               float SLEEP = 0.4,
               float SLEEP2 = 0.4):
        cdef unsigned int i, j, k, L, pos, _, num_root, length_cost, cursor, len_p, offset, len_pieces, ss
        cdef int rr
        cdef float min_score, score
        cdef tuple result
        cdef ndarray[np.uint8_t, ndim=1, cast=True] buc
        cdef ndarray[np.intp_t, ndim=1, cast=True] scores_sorted_index, scores2_sorted_index, scores3_sorted_index
        cdef ndarray[np.float64_t, ndim=1, cast=True] scores, scores2, scores3
        cdef bint p_in_all_words, violate
        cdef int (*num_roots)(LinguisticTokenizer, tuple, unsigned int, bint)
        
        if not return_all and use_cache:
            #with cache_lock:
            cache = self.cache
            if word in cache:
                return cache[word]
        combine = self.combine
        prefixes = self.prefixes
        suffixes = self.suffixes
        all_words = self.all_words
        very_exceptions = self.very_exceptions

        bucket1 = ()
        bucket2 = ()
        master_bucket = []
        fix_bucket1 = fix_bucket1
        fix_bucket2 = fix_bucket2
        orig = word
        seen = set()
        
        

        self.expand(word, 
                   word, 
                   bucket1, 
                   bucket2,
                   master_bucket,
                   fix_bucket1,
                   fix_bucket2,
                   '',
                   '',
                   0, 
                   max_N,
                   seen,
                   debug,
                   debug_slow,
                   SLEEP)
        if short:
            return master_bucket
        
        
        
        if debug:
            print('first final')
            if debug_slow:
                sleep(SLEEP2)

        single_words = self.single_words
        
        num_roots = self.num_roots


        to_be_cleans = set()
        if return_all:
            to_be_deleted = []
        #i = 0
        #for pieces, pos, _, violate in (master_bucket):
        #for pieces, pos, violate in (master_bucket):
        
        L = len(master_bucket)
        if L == 0:
            return ('',0,False)
        
        for k in range(L):
            pieces = master_bucket[k]
            pos = pieces[1]
            violate = pieces[2]
            pieces = pieces[0]
            len_pieces = len(pieces)
            if debug:
                print('first final 1.5', pieces, pos, violate)
                if debug_slow:
                    sleep(SLEEP2)
            if pos < len_pieces:
                p = pieces[pos]
                len_p = len(p)
                if len_p > 2 and p[len_p-1] == p[len_p-2] and not (p.endswith('ll') or p.endswith('ee')):

                    offset = 2
                    while len_p > offset + 1 and p[len_p-offset] == p[len_p-offset-1]:
                        offset += 1

                    reduced = p[:len_p-offset+1]
                    if reduced in all_words:
                        to_be_clean = reduced + 'e' 
                        if to_be_clean in all_words:
                            # clear other trailing 'e'
                            #master_bucket[i] = (tuple(reduced if j==pos else e for j, e in enumerate(pieces)), pos, _, violate)
                            master_bucket[k] = (tuple(reduced if j==pos else e for j, e in enumerate(pieces)), pos, violate)
                            if debug:
                                print('replace', (p, to_be_clean), master_bucket[k])
                                if debug_slow:
                                    sleep(SLEEP2)

                            to_be_cleans.add(to_be_clean)

            #i += 1
        if debug:
            print('first final 2')
            if debug_slow:
                sleep(SLEEP2)

        L = len(master_bucket)
        #for pieces, pos, ooo, violate in reversed(master_bucket):
        scores = np.empty(L, dtype=np.float64)
        vocabs = self.vocabs
        #for  in master_bucket:
        for k in range(L):
            pieces = master_bucket[k]
            pos = pieces[1]
            violate = pieces[2]
            pieces = pieces[0]
            len_pieces = len(pieces)
            if pos < len_pieces:
                p = pieces[pos]
                i = 0
                for to_be_clean in to_be_cleans:
                    if p == to_be_clean:
                        if debug:
                            print('del', master_bucket[k])
                            if debug_slow:
                                sleep(SLEEP2)
                        #del master_bucket[k]
                        if return_all:
                            to_be_deleted.append(k)
                        i = 1
                        break
                if i == 1:
                    scores[k] = 1000.
                    continue
                if p in very_exceptions:
                    pieces = pieces[:pos] + very_exceptions[p] + pieces[pos+1:]
                    master_bucket[k] = (pieces, pos, violate)

            rr = 0
            ss = 0
            #for p in pieces:
            _ = len(pieces)
            for i in range(_):
                p = pieces[i]
                p_in_all_words = p in all_words
                if p_in_all_words:
                    ss += 1
                    if i == pos:
                        rr = -2
                else:
                    if p in vocabs:
                        if i == pos:
                            rr = -1
                        ss += 1

            if ss == _:
                rr -= 10

            scores[k] = <int>(<int>int(violate))*100 -<int>((<float>ss / <float>_)*10.0) + rr


                                        # scores    # [ 3,5,6,7,8,1,1,1 ]
        scores_sorted_index = scores.argsort()      # [ 5,6,7,0,1,2,3,4 ]
        min_score = scores[scores_sorted_index[0]]  # 1

        if debug:
            print('first final 3', scores_sorted_index)
            if debug_slow:
                sleep(SLEEP2)


        i = 1
        while i < L:
            k = scores_sorted_index[i]
            score = scores[k]
            if score > min_score:
                break
            i += 1
        L = i
        scores2 = np.empty(L, dtype=np.float64)

        for i in range(L):
            k = scores_sorted_index[i]
            e = master_bucket[k]
            pos = e[1]
            violate = e[2]
            e = e[0]
            num_root = num_roots(self, e, pos, False)
            scores2[i] = (len(set(e))+max(1, num_root)*1.01 + (1.1 if num_root == 0 else 0) - int(all(c in single_words for c in e)))


        if debug:
            print('first final 4', scores2, L)
            if debug_slow:
                sleep(SLEEP2)
        scores2_sorted_index = scores2.argsort()     # [1,2,0]   =>  [6,7,5]
        if debug:
            print('first final 4.5', scores2_sorted_index)
            if debug_slow:
                sleep(SLEEP2)
        min_score = scores2[scores2_sorted_index[0]]  # 2

        if debug:
            print('first final 5', scores2, scores2_sorted_index)
            if debug_slow:
                sleep(SLEEP2)


        i = 1
        while i < L:
            k = scores2_sorted_index[i]
            score = scores2[k]
            if score > min_score:
                break
            i += 1
        L = i
        scores3 = np.empty(L, dtype=np.float64)
        if debug:
            print('first final 5.5', L)
            if debug_slow:
                sleep(SLEEP2)

        for j in range(L):
            e = master_bucket[scores_sorted_index[scores2_sorted_index[j]]]
            pos = e[1]
            violate = e[2]
            e = e[0]
            
            len_pieces = len(e)
            if debug:
                print('first final 5.2', pos, len_pieces)
                if debug_slow:
                    sleep(SLEEP2)
            
            if pos < len_pieces:
                i = len(e[pos])
            else:
                i = 0
            #if any(c.endswith('e') or c.endswith('o') for c in e):
            #    length_cost = min(lev(orig,''.join(c.rstrip('eo') for c in e)), length_cost)

            #orig = 'modal'
            #e = ('mode','al')
            pp = ''.join(e)
            buc = np.zeros(len(pp)+2, dtype=np.bool_)

            cursor = 0
            if len_pieces > 0:
                for _ in range(len_pieces - 1):
                    _e = e[_]
                    cursor += len(_e)
                    _ = len(_e)
                    if _ > 2:
                        last = _e[_-1]
                        if last == 'e' or last == 'o' or last == 'y':
                            buc[cursor-1] = 1
                    elif _ > 0:
                        last = _e[_-1]
                        if last == 'y':
                            buc[cursor-1] = 1

            if debug:
                print('first final 5.6', orig, pp)
                if debug_slow:
                    sleep(SLEEP2)


            scores3[j] = (levenshtein(orig.encode(), pp.encode(), buc)+1)*1000+(-i-1)*50+len_pieces

                                        # scores3   # [ -1, -2 ]

        scores3_sorted_index = scores3.argsort()     # [1,0]    =>  [7,6]
        if debug:
            print('first final 6', scores3, scores3_sorted_index)
            if debug_slow:
                sleep(SLEEP2)

        if debug:
            print('first final 7', scores2_sorted_index, scores3_sorted_index)
            if debug_slow:
                sleep(SLEEP2)
        if not return_all:
            result = master_bucket[scores_sorted_index[scores2_sorted_index[scores3_sorted_index[0]]]]
            #with cache_lock:
            #    cache[orig] = result
            if use_cache:
                cache[orig] = result

            if debug:
                return result, master_bucket, scores ,scores_sorted_index, scores2 ,scores2_sorted_index, scores3 ,scores3_sorted_index
            return result

        for k in reversed(to_be_deleted):
            del master_bucket[k]


        # handle very exceptions
        i = 0
        #for pieces, pos, _, violate in master_bucket:
        for pieces, pos, violate in master_bucket:
            len_pieces = len(pieces)
            if pos < len_pieces:
                p = pieces[pos]
                if p in very_exceptions:
                    new_pieces = pieces[:pos] + very_exceptions[p] + pieces[pos+1:]

                    #master_bucket[i] = (new_pieces, pos, _, violate)
                    master_bucket[i] = (new_pieces, pos, violate)
            i += 1
        
        _master_bucket = []

        if not return_all:
            vocabs = self.vocabs
            #sort_1 = self.sort_1
            '''
                # num. of parts
                len(set(e))+max(1, num_root)*1.01 + (1.1 if num_root == 0 else 0) - int(all(c in single_words for c in e)), 

            '''

            #master_bucket.sort(key=sort_1)
        else:
            for e, pos, violate in master_bucket:

                num_root = num_roots(self, e, pos, False)
                
                len_pieces = len(e)
                if pos < len_pieces:
                    root = e[pos]
                    #e_pos = root.rstrip('eo')
                else:
                    root = ''

                e_pos_in_all_words = root in all_words

                pp = ''.join(e)
                buc = np.zeros(len(pp)+2, dtype=np.bool_)

                cursor = 0
                for _ in range(len(e)):
                    _e = e[_]
                    cursor += len(_e)
                    i = len(_e)
                    if i > 2:
                        last = _e[i-1]
                        if last == 'e' or last == 'o' or last == 'y':
                            buc[cursor-1] = 1
                    elif _ > 0:
                        last = _e[_-1]
                        if last == 'y':
                            buc[cursor-1] = 1

                length_cost = levenshtein(orig.encode(), pp.encode(), buc)


                _master_bucket.append((e,pos,
                   (
                   int(violate),
                    # all in vocabulary

                    -sum(int(p in all_words or p in prefixes or p in suffixes) for p in e) / len(e),

                    # compositional validity
                    -int(e_pos_in_all_words)-int(e_pos_in_all_words or root in prefixes or root in suffixes),

                    # num. of parts
                    len(set(e))+max(1, num_root)*1.01 + (1.1 if num_root == 0 else 0) - int(all(c in single_words for c in e)), 

                    # length penalty
                    length_cost,

                    -len(root),#-max(len(p) for p in e),

                    #-int(e[pos] in all_words),
                    len(e)
                   ))
                )


        master_bucket = _master_bucket


        master_bucket.sort(key=lambda x:x[2])


        if return_all:
            return master_bucket


        for result in master_bucket:
            e,pos,ooo = result
            if pos < len(e) and (e[pos] in all_words or e[pos] in prefixes or e[pos] in suffixes):

                cache[orig] = result
                return result

        if len(master_bucket) > 0:
            result = master_bucket[0]
            cache[orig] = result
            return result
        '''
        
    def expand(self, 
               word, 
               orig=None, 
               tuple bucket1=None, 
               tuple bucket2=None, 
               master_bucket=None, 
               tuple fix_bucket1=None, 
               tuple fix_bucket2=None, 
               bint return_all=False, 
               #int morphed=0, 
               int N=0, 
               int max_N = 10,
               seen=None,
               bint debug=False,
               bint debug_slow=False,
               float SLEEP = 0.4,
               float SLEEP2 = 0.4):
        
        
        '''

    cdef void expand(self, 
                     str word, 
                     str orig, 
                     tuple bucket1, 
                     tuple bucket2, 
                     list master_bucket, 
                     tuple fix_bucket1, 
                     tuple fix_bucket2, 
                     str from_prefix_add,
                     str from_suffix_add,
                     unsigned int N, 
                     unsigned int max_N,
                     set seen,
                     bint debug,
                     bint debug_slow,
                     float SLEEP):
        cdef unsigned int i, j, k, pos, word_len, _
        cdef bint a_is_root, b_is_root, root, pre, suf, violate, bool1, bool2
        cdef tuple result
        #cdef bint (*check_violate)(tuple, int, bint)

        
        cdef bint (*check_violate)(LinguisticTokenizer, tuple, unsigned int, bint)
        cdef void (*expand)(LinguisticTokenizer, str, str, tuple, tuple, list, tuple, tuple, str, str, unsigned int, unsigned int, set, bint, bint, float)
        #cdef (tuple, double) (*combine)(LinguisticTokenizer, str, tuple, unsigned int, bint)
        
        
        
        N += 1
        if N > max_N:
            return 
        
        
        check_violate = self.check_violate
        combine = self.combine
        expand = self.expand
        
        if debug_slow:
            sleep(SLEEP)
        if debug:
            print('(%s) >> %s'%(N, str(fix_bucket1)+' ' +str(bucket1) + ' ' + word + ' ' + str(bucket2) + ' ' + str(fix_bucket2)))
            if debug_slow:
                sleep(SLEEP)
                
                

        very_exceptions = self.very_exceptions
        if word in very_exceptions:
            this = very_exceptions[word]
            expand(self,
                   this[0], 
                   word, 
                   bucket1, 
                   this[1:] + bucket2,
                   master_bucket,
                   fix_bucket1,
                   fix_bucket2,
                   '',
                   '',
                   N, 
                   max_N,
                   seen,
                   debug,
                   debug_slow,
                   SLEEP)
            return



        all_words = self.all_words
            
        cache = self.cache
        if word in cache:
            tks = cache[word][0]
            
            pos = len(bucket1)
            result = combine(orig, (bucket1 + tks + bucket2), pos, False)
            pos = result[1]
            result = result[0]

            if fix_bucket1 is not None:
                result = fix_bucket1 + result
                pos += len(fix_bucket1)
            if fix_bucket2 is not None:
                result = result + fix_bucket2
                
                
                
            master_bucket.append((result, 
                                  pos,  
                                  #morphed,#not (pos>=0 and result[pos] in all_words) or morphed, 
                                  check_violate(self, result, pos, False)))
            if debug:
                print('append', result, pos)
                print('****',(result, pos, (bucket1 + tks + bucket2)))
                if debug_slow:
                    sleep(SLEEP)

            return


        if debug:
            print('ckpt 1')
            if debug_slow:
                sleep(SLEEP)

        
        
        if word == '':
            pos = 9999

            result = combine(orig,bucket1 + bucket2, pos, False)
            pos = result[1]
            result = result[0]
            
            

            if self.check_seen(orig, bucket1,word,bucket2,fix_bucket1,fix_bucket2,pos,seen):
                return
            if debug:
                print('ckpt 1.2')
                if debug_slow:
                    sleep(SLEEP)


            if fix_bucket1 is not None:
                result = fix_bucket1 + result
                pos += len(fix_bucket1)
            if fix_bucket2 is not None:
                result = result + fix_bucket2
            master_bucket.append((result, 
                                  pos, 
                                  #morphed,#not (pos>=0 and result[pos] in all_words) or morphed, 
                                  check_violate(self, result, pos, False)))

            if debug:
                print('append', result, pos)
                print('****',(result, pos, (bucket1 + (word,) + bucket2)))
                if debug_slow:
                    sleep(SLEEP)
            #if not first:
            #    return 
            #else:
            #    return ([''],0,0)
            return 

        #check_seen = self.check_seen
        word_len = len(word)
        segments_left = [word[:_] for _ in range(1,word_len)]
        segments_right = [word[word_len-_:] for _ in range(1,word_len)]
        pos = len(bucket1)
        result = combine(orig, (bucket1 + (word,) + bucket2), pos, False)
        pos = result[1]
        result = result[0]
        
        if self.check_seen(orig,
                           bucket1,
                           word,
                           bucket2,
                           fix_bucket1,
                           fix_bucket2,
                           pos,
                           seen):
            return

        if debug:
            print('ckpt 2', result, pos, word)
            if debug_slow:
                sleep(SLEEP)
        prefixes = self.prefixes
        suffixes = self.suffixes
        root = word in all_words
        if (word in prefixes and any(e=='' for e in prefixes[word])) or (word in suffixes and '' in suffixes[word]) or root:

            if debug:
                print('ckpt 2.5')
                if debug_slow:
                    sleep(SLEEP)
            if fix_bucket1 is not None:
                result = fix_bucket1 + result
                pos += len(fix_bucket1)
            if fix_bucket2 is not None:
                result = result + fix_bucket2
            master_bucket.append((result, 
                                  pos,  
                                  #morphed,#not (pos>=0 and result[pos] in all_words) or morphed, 
                                  check_violate(self, result, pos, False)))
            if debug:
                print('append', result, pos)

                print('****',(result, pos, (bucket1 + (word,) + bucket2)))
                if debug_slow:
                    sleep(SLEEP)
            #if not first:
            #    return 
            #else:
            #    return (bucket1+(word,) + bucket2,0 if root else -1,0)
            
            
            return 

        if debug:
            print('ckpt 3')
            if debug_slow:
                sleep(SLEEP)
            
        if word_len > 2:
            last = word[word_len-1]
            if last == word[word_len-2]:
            

                expand(self,
                       segments_left[word_len-2], 
                       orig, 
                       bucket1, 
                       bucket2,
                       master_bucket,
                       fix_bucket1,
                       fix_bucket2,
                       '',
                       '',
                       N, 
                       max_N,
                       seen,
                       debug,
                       debug_slow,
                       SLEEP)
        
        if debug:
            print('ckpt 4')
            if debug_slow:
                sleep(SLEEP)
        
        if word_len > 2:
            if word[0] == word[1]:
                expand(self,
                       segments_right[word_len-2], 
                       orig, 
                       bucket1, 
                       bucket2,
                       master_bucket,
                       fix_bucket1,
                       fix_bucket2,
                       '',
                       '',
                       N, 
                       max_N,
                       seen,
                       debug,
                       debug_slow,
                       SLEEP)
        if debug:
            print('ckpt 5')
            if debug_slow:
                sleep(SLEEP)

        '''
        for p in prefixes:
            if word.startswith(p) and p not in bucket1:
                '''
        ###########################
        ########## DEBUG ##########
        ###########################
        if debug:
            prefixes_spotted = []
            for _ in range(1,word_len):
                p = word[:_]
                if p in prefixes and p not in bucket1:
                    prefixes_spotted.append(p)
            print('prefixes_spotted:', prefixes_spotted)
            if debug_slow:
                sleep(SLEEP)
                
                
            suffixes_spotted = []
            for _ in range(1,word_len):
                s = word[word_len-_:]
                if s in suffixes and s not in bucket2:

                    suffixes_spotted.append(s)
                if len(bucket2) > 0 and s in prefixes:
                    suffixes_spotted.append(s)
            print('suffixes_spotted:', suffixes_spotted)
            if debug_slow:
                sleep(SLEEP)
                
        if word_len > 2:
            len_bucket1 = len(bucket1)
            len_bucket2 = len(bucket2)
            bool1 = len_bucket1 > 0
            bool2 = len_bucket2 > 0
            if bool1:
                bucket1_last_one = bucket1[len_bucket1-1]
            if bool2:
                bucket2_first_one = bucket2[0]
            expand_exceptions = self.expand_exceptions
            expand_exceptions_reverse = self.expand_exceptions_reverse
            for j in range(word_len-2):
                _len = j+1
                p = segments_left[j]
                if debug:
                    print('assume prefix:', p)
                    if debug_slow:
                        sleep(SLEEP)
                pre = p in prefixes
                #suf = p in suffixes

                if pre: # and ((not bool1) or p != bucket1_last_one):
                    
                    # Here has a prefix found
                    if bool1 and bucket1_last_one in expand_exceptions:
                        tt = expand_exceptions[bucket1_last_one][0]
                        if tt is not None and (p in tt or len(tt) == 0):
                            continue

                    '''
                    for add in prefixes[p]:
                        if debug:
                            print('ckpt 6')
                            if debug_slow:
                                sleep(SLEEP)
                        expand(self,
                               word[_len:] + add, 
                               orig, 
                               bucket1 + (p,), 
                               bucket2,
                               master_bucket,
                               fix_bucket1,
                               fix_bucket2,
                               add,
                               '',
                               N, 
                               max_N,
                               seen,
                               debug,
                               debug_slow,
                               SLEEP)
                    '''
                    expand(self,
                           word[_len:], 
                           orig, 
                           bucket1 + (p,), 
                           bucket2,
                           master_bucket,
                           fix_bucket1,
                           fix_bucket2,
                           '',
                           '',
                           N, 
                           max_N,
                           seen,
                           debug,
                           debug_slow,
                           SLEEP)


                s = segments_right[j]
                #pre = s in prefixes
                suf = s in suffixes
                if debug:
                    print('assume suffix:', s)
                    if debug_slow:
                        sleep(SLEEP)
                if suf and (s != from_suffix_add or from_suffix_add == 'y'): # and ((not bool2) or s != bucket2_first_one):
                    # Here has a prefix found
                    if bool2 and bucket2_first_one in expand_exceptions_reverse:
                        tt = expand_exceptions_reverse[bucket2_first_one][0]
                        if tt is not None and (s in tt or len(tt) == 0):
                            continue

                    remain = segments_left[word_len-2-j]  #word_len-2-(j+1)

                    if debug:
                        print('j',j)
                        print('word_len',word_len)
                        print(segments_left)
                        print('remain', remain)
                        if debug_slow:
                            sleep(SLEEP)
                    for add in suffixes[s]:
                        _word = remain + add
                        if debug:

                            print('suffix:', _word)



                            print('ckpt 7')
                            #print('__word:', __word, 'len:', word_len-_len)
                            print('__word:', remain, 'len:', word_len-_len)
                            if debug_slow:
                                sleep(SLEEP)
                            
                            
                            
                        
                        expand(self,
                               _word, 
                               orig, 
                               bucket1, 
                               (s,) + bucket2,
                               master_bucket,
                               fix_bucket1,
                               fix_bucket2,
                               '',
                               add,
                               N, 
                               max_N,
                               seen,
                               debug,
                               debug_slow,
                               SLEEP)

                        #if __word is not None:
                        #if remain is not None:
                        __word_len = word_len-_len #len(__word) 
                        #if __word_len > 2 and __word[__word_len-1] == __word[__word_len-2]:
                        if __word_len > 2 and remain[__word_len-1] == remain[__word_len-2]:
                            if debug:
                                print('ckpt 7.5')
                                if debug_slow:
                                    sleep(SLEEP)

                            expand(self,
                                   word[:max(0,word_len-_len-1)] + add, 
                                   orig, 
                                   bucket1, 
                                   (s,) + bucket2,
                                   master_bucket,
                                   fix_bucket1,
                                   fix_bucket2,
                                   '',
                                   add,
                                   N, 
                                   max_N,
                                   seen,
                                   debug,
                                   debug_slow,
                                   SLEEP)


                if len(bucket2) > 0 and (s != from_suffix_add or from_suffix_add == 'y') and s in prefixes:
                    if debug:
                        print('ckpt 8')
                        if debug_slow:
                            sleep(SLEEP)
                            
                    expand(self,
                           word[:word_len-_len], 
                           orig, 
                           bucket1, 
                           (s,) + bucket2,
                           master_bucket,
                           fix_bucket1,
                           fix_bucket2,
                           '',
                           '',
                           N, 
                           max_N,
                           seen,
                           debug,
                           debug_slow,
                           SLEEP)

        if debug:
            print('ckpt 9')
            if debug_slow:
                sleep(SLEEP)


        pos = 9999 if not word else len(bucket1)

        result = combine( orig, 
                               (bucket1 + (word,) + bucket2) if word else (bucket1 + bucket2), 
                               pos, 
                               False)
        pos = result[1]
        result = result[0]

        if debug:
            print('ckpt 9.1')
        if fix_bucket1 is not None:
            result = fix_bucket1 + result
            pos += len(fix_bucket1)
        if fix_bucket2 is not None:
            result = result + fix_bucket2
        master_bucket.append((result, 
                              pos, 
                              #morphed,#not (pos>=0 and result[pos] in all_words) or morphed, 
                              check_violate(self, result, pos, False)))
        if debug:
            print('append', result, pos)
            print('****',(result, pos, (bucket1 + (word,) + bucket2)))
            if debug_slow:
                sleep(SLEEP)
                
                
                
                
        if debug:
            print('ckpt 9.5')
            if debug_slow:
                sleep(SLEEP)
        root_only = self.root_only
        if debug:
            compound_splotted = []
            for _ in range(1,word_len-1):
                a,b = word[:_], word[_:]
                a_is_root = a in root_only
                b_is_root = b in root_only
                if a_is_root and b_is_root: # and a not in prefixes and b not in suffixes:# and a not in combine_exceptions and b not in combine_exceptions:
                    result = combine(orig, bucket1+(a,b)+bucket2, len(bucket1), False)
                    pos = result[1]
                    result = result[0]
                    
                    if fix_bucket1 is not None:
                        result = fix_bucket1 + result
                        pos += len(fix_bucket1)
                    if fix_bucket2 is not None:
                        result = result + fix_bucket2
                    if check_violate(self, result, pos, False) :
                        continue
                    
                elif a_is_root: #a in all_words and a not in prefixes:# and a not in combine_exceptions:

                    pass
                elif b_is_root: #b in all_words and b not in suffixes:# and b not in combine_exceptions:

                    pass
                else:
                    continue
                compound_splotted.append((a,b))
                
            print('compund found:', compound_splotted)
            if debug_slow:
                sleep(SLEEP)
        if debug:
            print('ckpt 9.8')
            if debug_slow:
                sleep(SLEEP)
                
                
        for _ in range(1,word_len):
            a,b = word[:_], word[_:]
            
            a_is_root = a in root_only
            b_is_root = b in root_only
            
            if a_is_root and b_is_root: # and a not in prefixes and b_not_suffix:# and a not in combine_exceptions and b not in combine_exceptions:
                temp = bucket1+(a,b)+bucket2
                _  = len(bucket1)
                result = combine(orig, temp, _, False)
                pos = result[1]
                result = result[0]
                
                if debug:
                    print(result, pos)
                    if debug_slow:
                        sleep(SLEEP)
                #if not :# and (pos>=0 and result[pos] in all_words):
                if fix_bucket1 is not None:
                    result = fix_bucket1 + result
                    pos += len(fix_bucket1)
                if fix_bucket2 is not None:
                    result = result + fix_bucket2
                master_bucket.append((result, 
                                      pos, 
                                      #morphed,#not (pos>=0 and result[pos] in all_words) or morphed, 
                                      check_violate(self, result, pos, False)))
                
                if debug:
                    print('append', result, pos)
                
                result = combine(orig, temp, _+1, False)
                pos = result[1]
                result = result[0]
                
                if debug:
                    print(result, pos)
                    if debug_slow:
                        sleep(SLEEP)
                #if not :# and (pos>=0 and result[pos] in all_words):
                if fix_bucket1 is not None:
                    result = fix_bucket1 + result
                    pos += len(fix_bucket1)
                if fix_bucket2 is not None:
                    result = result + fix_bucket2
                master_bucket.append((result, 
                                      pos, 
                                      #morphed,#not (pos>=0 and result[pos] in all_words) or morphed, 
                                      check_violate(self, result, pos, False)))


                if debug:
                    print('append', result, pos)
                    print('ckpt 9.9', bucket1+(a,b)+bucket2)
                    if debug_slow:
                        sleep(SLEEP)
            else:
                
                if a_is_root:# and a not in combine_exceptions:

                    if debug:
                        print('ckpt 10')
                        if debug_slow:
                            sleep(SLEEP)
                    expand(self,
                           b, 
                           orig, 
                           (), 
                           bucket2,
                           master_bucket,
                           (() if fix_bucket1 is None else fix_bucket1)+bucket1+(a,),
                           fix_bucket2,
                           '',
                           '',
                           N, 
                           max_N,
                           seen,
                           debug,
                           debug_slow,
                           SLEEP)

                if b_is_root:# and b not in combine_exceptions:

                    if debug:
                        print('ckpt 11')
                        if debug_slow:
                            sleep(SLEEP)
                            
                    expand(self,
                           a, 
                           orig, 
                           bucket1, 
                           (),
                           master_bucket,
                           fix_bucket1,
                           (b,)+bucket2 + (() if fix_bucket2 is None else fix_bucket2),
                           '',
                           '',
                           N, 
                           max_N,
                           seen,
                           debug,
                           debug_slow,
                           SLEEP)

                    #return [result,-2]

        if debug:
            print('end expand')
            if debug_slow:
                sleep(SLEEP)
            
        
    def print_expand(self, e, print_if_oov=False):
        prefixes = self.prefixes
        suffixes = self.suffixes
        all_words = self.all_words
        splitted = self.tokenize(e)
        oov = any((False if i==splitted[1] and c in all_words else 
                                (False if (True or i < splitted[1] or i > splitted[1]) and (c in prefixes or c in all_words or c in suffixes) else True)
                               ) for i, c in enumerate(splitted[0]))
        if not print_if_oov or oov:
            print(e+' '+'( '+ ','.join([('*' if i==splitted[1] else '')+(colored(c,'blue') if i==splitted[1] and c in all_words else 
                        (c if (True or i < splitted[1] or i > splitted[1]) and (c in prefixes or c in all_words or c in suffixes) else colored(c, 'red'))
                       ) for i, c in enumerate(splitted[0])])+') >> '+ str(splitted[1:]))
        return oov, splitted
        
        
from termcolor import colored

'''
from LinguisticTokenizer import LinguisticTokenizer
tokenizer = LinguisticTokenizer()
tokenizer.init()
'''