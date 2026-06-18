#!/usr/bin/perl
use strict;
use IO::CaptureOutput qw(capture qxx qxy);
use Statistics::Descriptive;
use Cwd;

my $nIterations = 1;

open( FILE, '>epcTimes.csv' );
print FILE "#sTime\tnodes\trTime\trTDev\n";

my @nodes = ( 1,2,3,4,5,6,7, 8, 12, 14);
my @simTime = (5, 10);


# Configure and complite first the program to avoid counting compilation time as running time
my $launch = "CXXFLAGS=\"-O3 -w\" ./ns3 configure -d optimized --enable-static --enable-examples --enable-modules=lte";
my $out, my $err;
capture { system($launch ) } \$out, \$err;
$launch = "./ns3 run \'lena-profiling --simTime=0.1 --nUe=1 --nEnb=1 --nFloors=0\'";

foreach my $time (@simTime)
{
         foreach my $node (@nodes)
         {
            my $timeStats = Statistics::Descriptive::Full->new();
            for ( my $iteration = 0 ; $iteration < $nIterations ; $iteration++ )
            {
               $launch = "time -f \"real%E\" ./ns3 run 'lena-simple-epc --simTime=$time --numberOfNodes=$node'";
               print "$launch\n";
               capture { system($launch ) } \$out, \$err;
               $err =~ /real(.+):(.+)/;
               my $minutes = $1;
               my $seconds = $minutes * 60 + $2;
               $timeStats->add_data($seconds);
            }
            print FILE "$time\t$node\t";
            print FILE $timeStats->mean() . "\t"
              . $timeStats->standard_deviation() . "\n";
         }
}
