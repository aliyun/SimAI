#!/usr/bin/python


# output xml format:
# <pages>
# <page url="xx"><prev url="yyy">zzz</prev><next url="hhh">lll</next><fragment>file.frag</fragment></page>
# ...
# </pages>

import pickle
import os
import codecs

def dump_pickles(out, dirname, filename, path):
    f = open(os.path.join(dirname, filename), 'r')
    data = pickle.load(f)
    fragment_file = codecs.open(data['current_page_name'] + '.frag', mode='w', encoding='utf-8')
    fragment_file.write(data['body'])
    fragment_file.close()
    out.write('  <page url="%s">\n' % path)
    out.write('    <fragment>%s.frag</fragment>\n' % data['current_page_name'])
    if data['prev'] is not None:
        out.write('    <prev url="%s">%s</prev>\n' % 
                  (os.path.normpath(os.path.join(path, data['prev']['link'])), 
                   data['prev']['title']))
    if data['next'] is not None:
        out.write('    <next url="%s">%s</next>\n' % 
                  (os.path.normpath(os.path.join(path, data['next']['link'])), 
                   data['next']['title']))
    out.write('  </page>\n')
    f.close()
    if data['next'] is not None:
        next_path = os.path.normpath(os.path.join(path, data['next']['link']))
        next_filename = os.path.basename(next_path) + '.fpickle'
        dump_pickles(out, dirname, next_filename, next_path)
    return

import sys

sys.stdout.write('<pages>\n')
dump_pickles(sys.stdout, os.path.dirname(sys.argv[1]), os.path.basename(sys.argv[1]), '/')
sys.stdout.write('</pages>')
