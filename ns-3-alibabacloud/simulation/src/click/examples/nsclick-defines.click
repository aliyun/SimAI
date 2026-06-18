define($OUTPUT "This is a default value for OUTPUT");
define($UNDEFINED "UNDEFINED is an undefined variable");

// It is mandatory to use an IPRouteTable element with ns-3-click
rt :: LinearIPLookup ();
Idle -> rt;

output :: Message($OUTPUT);
undefined :: Message($UNDEFINED);
