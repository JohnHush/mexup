import os
import sys
import fnmatch

def find_recursive( path , pattern ):
    return_list = []
    for dirpath, dirnames, filenames in os.walk(path):
        if not filenames:
            continue

        pythonic_files = fnmatch.filter(filenames, pattern)
        if pythonic_files:
            for file in pythonic_files:
                full_name = os.path.abspath(os.path.join( dirpath, file ))
                # print( full_name )
                # print('{}/{}'.format(dirpath, file))
                return_list.append( full_name )

    return return_list

def run():
    dir_name = sys.argv[1]
    rl = find_recursive( dir_name , '*.cue' )
    # os.system( 'iconv --help' )

    for ff in rl:
        gg = ff+'2'
        # print( ff )
        print( gg )
        os.system( 'iconv -f GBK -t UTF-8 "%s" > "%s"' %( ff, gg ) )
        os.system( 'mv "%s" "%s"' %( gg, ff ) )
        # os.system( 'rm "%s"'%(gg) )

run()