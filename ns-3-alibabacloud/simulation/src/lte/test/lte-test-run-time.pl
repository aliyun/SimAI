#!/usr/bin/perl
use strict;
use IO::CaptureOutput qw(capture qxx qxy);
use Statistics::Descriptive;

my $nIterations = 1;

open( FILE, '>lteTimes.csv' );
print FILE "#sTime\tnFloors\tnEnb\tnUe\trTime\trTDev\n";

my @nUe = (1, 5, 10, 15, 20, 25, 30);
my @nEnb = (1, 2, 4, 6, 8, 12, 14, 18, 22);
my @nFloors = (0, 1);
my @simTime = (5, 10);

# Configure and complite first the program to avoid counting compilation time as running time
my $launch = "CXXFLAGS=\"-O3 -w\" ./ns3 configure -d optimized --enable-static --enable-examples --enable-modules=lte";
my $out;
my $err;
capture { system($launch ) } \$out, \$err;
$launch = "./ns3 run \'lena-profiling --simTime=0.1 --nUe=1 --nEnb=1 --nFloors=0\'";
capture { system($launch ) } \$out, \$err;

foreach my $time (@simTime)
{
   foreach my $floor (@nFloors)
   {
      foreach my $enb (@nEnb)
      {
         foreach my $ue (@nUe)
         {
            my $timeStats = Statistics::Descriptive::Full->new();
            for ( my $iteration = 0 ; $iteration < $nIterations ; $iteration++ )
            {
               $launch = "time -f \"real%E\" ./ns3 run \'lena-profiling --simTime=$time --nUe=$ue --nEnb=$enb --nFloors=$floor\'";
               print "$launch\n";
               capture { system($launch ) } \$out, \$err;
               $err =~ /real(.+):(.+)/;
               my $minutes = $1;
               my $seconds = $minutes * 60 + $2;
               $timeStats->add_data($seconds);
            }
            print FILE "$time\t$floor\t$enb\t$ue\t";
            print FILE $timeStats->mean() . "\t"
              . $timeStats->standard_deviation() . "\n";
         }
      }
   }
}
