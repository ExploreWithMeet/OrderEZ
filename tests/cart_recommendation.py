from utils.convex import call_convex
from schema.convex_schema import ConvexRequest

#

if __name__ == "__main__":
    # CALLING OLD RULES
    # CALLING NEW ORDER HISTORY
    data = call_convex(ConvexRequest())
    # PREPROCESSING
    # APPLYING APRIORI
    # COMBINING OLD & NEW RULES
    # POSTING TO CONVEX
