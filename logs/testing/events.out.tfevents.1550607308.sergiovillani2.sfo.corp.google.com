       �K"	   ��Abrain.Event:2�ːNM�      ��	"�>��A"��
t
input/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
�
1layer_1/weights1/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_1/weights1*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_1/weights1/Initializer/random_uniform/minConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7��*
dtype0*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/maxConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7�?*
dtype0*
_output_shapes
: 
�
9layer_1/weights1/Initializer/random_uniform/RandomUniformRandomUniform1layer_1/weights1/Initializer/random_uniform/shape*

seed *
T0*#
_class
loc:@layer_1/weights1*
seed2 *
dtype0*
_output_shapes

:
�
/layer_1/weights1/Initializer/random_uniform/subSub/layer_1/weights1/Initializer/random_uniform/max/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/mulMul9layer_1/weights1/Initializer/random_uniform/RandomUniform/layer_1/weights1/Initializer/random_uniform/sub*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
�
+layer_1/weights1/Initializer/random_uniformAdd/layer_1/weights1/Initializer/random_uniform/mul/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
layer_1/weights1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1*
	container 
�
layer_1/weights1/AssignAssignlayer_1/weights1+layer_1/weights1/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
�
layer_1/weights1/readIdentitylayer_1/weights1*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
�
!layer_1/biases1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
layer_1/biases1
VariableV2*"
_class
loc:@layer_1/biases1*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
layer_1/biases1/AssignAssignlayer_1/biases1!layer_1/biases1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
z
layer_1/biases1/readIdentitylayer_1/biases1*
_output_shapes
:*
T0*"
_class
loc:@layer_1/biases1
�
layer_1/MatMulMatMulinput/Placeholderlayer_1/weights1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
j
layer_1/addAddlayer_1/MatMullayer_1/biases1/read*
T0*'
_output_shapes
:���������
S
layer_1/ReluRelulayer_1/add*
T0*'
_output_shapes
:���������
�
1layer_2/weights2/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_2/weights2*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_2/weights2/Initializer/random_uniform/minConst*#
_class
loc:@layer_2/weights2*
valueB
 *׳]�*
dtype0*
_output_shapes
: 
�
/layer_2/weights2/Initializer/random_uniform/maxConst*#
_class
loc:@layer_2/weights2*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
�
9layer_2/weights2/Initializer/random_uniform/RandomUniformRandomUniform1layer_2/weights2/Initializer/random_uniform/shape*

seed *
T0*#
_class
loc:@layer_2/weights2*
seed2 *
dtype0*
_output_shapes

:
�
/layer_2/weights2/Initializer/random_uniform/subSub/layer_2/weights2/Initializer/random_uniform/max/layer_2/weights2/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@layer_2/weights2
�
/layer_2/weights2/Initializer/random_uniform/mulMul9layer_2/weights2/Initializer/random_uniform/RandomUniform/layer_2/weights2/Initializer/random_uniform/sub*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
+layer_2/weights2/Initializer/random_uniformAdd/layer_2/weights2/Initializer/random_uniform/mul/layer_2/weights2/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
layer_2/weights2
VariableV2*#
_class
loc:@layer_2/weights2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
layer_2/weights2/AssignAssignlayer_2/weights2+layer_2/weights2/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
layer_2/weights2/readIdentitylayer_2/weights2*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
!layer_2/biases2/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
layer_2/biases2
VariableV2*"
_class
loc:@layer_2/biases2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
layer_2/biases2/AssignAssignlayer_2/biases2!layer_2/biases2/Initializer/zeros*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
z
layer_2/biases2/readIdentitylayer_2/biases2*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
layer_2/MatMulMatMullayer_1/Relulayer_2/weights2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
j
layer_2/addAddlayer_2/MatMullayer_2/biases2/read*
T0*'
_output_shapes
:���������
S
layer_2/ReluRelulayer_2/add*
T0*'
_output_shapes
:���������
�
1layer_3/weights3/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*#
_class
loc:@layer_3/weights3*
valueB"      
�
/layer_3/weights3/Initializer/random_uniform/minConst*#
_class
loc:@layer_3/weights3*
valueB
 *׳]�*
dtype0*
_output_shapes
: 
�
/layer_3/weights3/Initializer/random_uniform/maxConst*#
_class
loc:@layer_3/weights3*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
�
9layer_3/weights3/Initializer/random_uniform/RandomUniformRandomUniform1layer_3/weights3/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_3/weights3*
seed2 
�
/layer_3/weights3/Initializer/random_uniform/subSub/layer_3/weights3/Initializer/random_uniform/max/layer_3/weights3/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@layer_3/weights3
�
/layer_3/weights3/Initializer/random_uniform/mulMul9layer_3/weights3/Initializer/random_uniform/RandomUniform/layer_3/weights3/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
+layer_3/weights3/Initializer/random_uniformAdd/layer_3/weights3/Initializer/random_uniform/mul/layer_3/weights3/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
layer_3/weights3
VariableV2*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:*
dtype0*
_output_shapes

:
�
layer_3/weights3/AssignAssignlayer_3/weights3+layer_3/weights3/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
�
layer_3/weights3/readIdentitylayer_3/weights3*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
!layer_3/biases3/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_3/biases3*
valueB*    
�
layer_3/biases3
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3
�
layer_3/biases3/AssignAssignlayer_3/biases3!layer_3/biases3/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
z
layer_3/biases3/readIdentitylayer_3/biases3*
_output_shapes
:*
T0*"
_class
loc:@layer_3/biases3
�
layer_3/MatMulMatMullayer_2/Relulayer_3/weights3/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
j
layer_3/addAddlayer_3/MatMullayer_3/biases3/read*
T0*'
_output_shapes
:���������
S
layer_3/ReluRelulayer_3/add*'
_output_shapes
:���������*
T0
�
0output/weights4/Initializer/random_uniform/shapeConst*"
_class
loc:@output/weights4*
valueB"      *
dtype0*
_output_shapes
:
�
.output/weights4/Initializer/random_uniform/minConst*"
_class
loc:@output/weights4*
valueB
 *qĜ�*
dtype0*
_output_shapes
: 
�
.output/weights4/Initializer/random_uniform/maxConst*"
_class
loc:@output/weights4*
valueB
 *qĜ?*
dtype0*
_output_shapes
: 
�
8output/weights4/Initializer/random_uniform/RandomUniformRandomUniform0output/weights4/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*"
_class
loc:@output/weights4*
seed2 
�
.output/weights4/Initializer/random_uniform/subSub.output/weights4/Initializer/random_uniform/max.output/weights4/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@output/weights4
�
.output/weights4/Initializer/random_uniform/mulMul8output/weights4/Initializer/random_uniform/RandomUniform.output/weights4/Initializer/random_uniform/sub*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
*output/weights4/Initializer/random_uniformAdd.output/weights4/Initializer/random_uniform/mul.output/weights4/Initializer/random_uniform/min*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
output/weights4
VariableV2*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4*
	container *
shape
:
�
output/weights4/AssignAssignoutput/weights4*output/weights4/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
~
output/weights4/readIdentityoutput/weights4*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
 output/biases4/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
output/biases4
VariableV2*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:*
dtype0*
_output_shapes
:
�
output/biases4/AssignAssignoutput/biases4 output/biases4/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
w
output/biases4/readIdentityoutput/biases4*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
�
output/MatMulMatMullayer_3/Reluoutput/weights4/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
g

output/addAddoutput/MatMuloutput/biases4/read*
T0*'
_output_shapes
:���������
s
cost/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
{
cost/SquaredDifferenceSquaredDifference
output/addcost/Placeholder*'
_output_shapes
:���������*
T0
[

cost/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
s
	cost/MeanMeancost/SquaredDifference
cost/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
train/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
}
,train/gradients/cost/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
&train/gradients/cost/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/cost/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
z
$train/gradients/cost/Mean_grad/ShapeShapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
#train/gradients/cost/Mean_grad/TileTile&train/gradients/cost/Mean_grad/Reshape$train/gradients/cost/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
|
&train/gradients/cost/Mean_grad/Shape_1Shapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/cost/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
n
$train/gradients/cost/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
#train/gradients/cost/Mean_grad/ProdProd&train/gradients/cost/Mean_grad/Shape_1$train/gradients/cost/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
p
&train/gradients/cost/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
%train/gradients/cost/Mean_grad/Prod_1Prod&train/gradients/cost/Mean_grad/Shape_2&train/gradients/cost/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
j
(train/gradients/cost/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
&train/gradients/cost/Mean_grad/MaximumMaximum%train/gradients/cost/Mean_grad/Prod_1(train/gradients/cost/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
'train/gradients/cost/Mean_grad/floordivFloorDiv#train/gradients/cost/Mean_grad/Prod&train/gradients/cost/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
#train/gradients/cost/Mean_grad/CastCast'train/gradients/cost/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
&train/gradients/cost/Mean_grad/truedivRealDiv#train/gradients/cost/Mean_grad/Tile#train/gradients/cost/Mean_grad/Cast*
T0*'
_output_shapes
:���������
{
1train/gradients/cost/SquaredDifference_grad/ShapeShape
output/add*
T0*
out_type0*
_output_shapes
:
�
3train/gradients/cost/SquaredDifference_grad/Shape_1Shapecost/Placeholder*
T0*
out_type0*
_output_shapes
:
�
Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/cost/SquaredDifference_grad/Shape3train/gradients/cost/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2train/gradients/cost/SquaredDifference_grad/scalarConst'^train/gradients/cost/Mean_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
�
/train/gradients/cost/SquaredDifference_grad/mulMul2train/gradients/cost/SquaredDifference_grad/scalar&train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/subSub
output/addcost/Placeholder'^train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/mul_1Mul/train/gradients/cost/SquaredDifference_grad/mul/train/gradients/cost/SquaredDifference_grad/sub*'
_output_shapes
:���������*
T0
�
/train/gradients/cost/SquaredDifference_grad/SumSum1train/gradients/cost/SquaredDifference_grad/mul_1Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
3train/gradients/cost/SquaredDifference_grad/ReshapeReshape/train/gradients/cost/SquaredDifference_grad/Sum1train/gradients/cost/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/Sum_1Sum1train/gradients/cost/SquaredDifference_grad/mul_1Ctrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
5train/gradients/cost/SquaredDifference_grad/Reshape_1Reshape1train/gradients/cost/SquaredDifference_grad/Sum_13train/gradients/cost/SquaredDifference_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
/train/gradients/cost/SquaredDifference_grad/NegNeg5train/gradients/cost/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
<train/gradients/cost/SquaredDifference_grad/tuple/group_depsNoOp0^train/gradients/cost/SquaredDifference_grad/Neg4^train/gradients/cost/SquaredDifference_grad/Reshape
�
Dtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependencyIdentity3train/gradients/cost/SquaredDifference_grad/Reshape=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/cost/SquaredDifference_grad/Reshape*'
_output_shapes
:���������
�
Ftrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/cost/SquaredDifference_grad/Neg=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/cost/SquaredDifference_grad/Neg*'
_output_shapes
:���������
r
%train/gradients/output/add_grad/ShapeShapeoutput/MatMul*
T0*
out_type0*
_output_shapes
:
q
'train/gradients/output/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
5train/gradients/output/add_grad/BroadcastGradientArgsBroadcastGradientArgs%train/gradients/output/add_grad/Shape'train/gradients/output/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
#train/gradients/output/add_grad/SumSumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency5train/gradients/output/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
'train/gradients/output/add_grad/ReshapeReshape#train/gradients/output/add_grad/Sum%train/gradients/output/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
%train/gradients/output/add_grad/Sum_1SumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency7train/gradients/output/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
)train/gradients/output/add_grad/Reshape_1Reshape%train/gradients/output/add_grad/Sum_1'train/gradients/output/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
0train/gradients/output/add_grad/tuple/group_depsNoOp(^train/gradients/output/add_grad/Reshape*^train/gradients/output/add_grad/Reshape_1
�
8train/gradients/output/add_grad/tuple/control_dependencyIdentity'train/gradients/output/add_grad/Reshape1^train/gradients/output/add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/output/add_grad/Reshape*'
_output_shapes
:���������
�
:train/gradients/output/add_grad/tuple/control_dependency_1Identity)train/gradients/output/add_grad/Reshape_11^train/gradients/output/add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/add_grad/Reshape_1*
_output_shapes
:
�
)train/gradients/output/MatMul_grad/MatMulMatMul8train/gradients/output/add_grad/tuple/control_dependencyoutput/weights4/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
+train/gradients/output/MatMul_grad/MatMul_1MatMullayer_3/Relu8train/gradients/output/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
3train/gradients/output/MatMul_grad/tuple/group_depsNoOp*^train/gradients/output/MatMul_grad/MatMul,^train/gradients/output/MatMul_grad/MatMul_1
�
;train/gradients/output/MatMul_grad/tuple/control_dependencyIdentity)train/gradients/output/MatMul_grad/MatMul4^train/gradients/output/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*<
_class2
0.loc:@train/gradients/output/MatMul_grad/MatMul
�
=train/gradients/output/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/output/MatMul_grad/MatMul_14^train/gradients/output/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/output/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_3/Relu_grad/ReluGradReluGrad;train/gradients/output/MatMul_grad/tuple/control_dependencylayer_3/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_3/add_grad/ShapeShapelayer_3/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_3/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/add_grad/Shape(train/gradients/layer_3/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_3/add_grad/SumSum*train/gradients/layer_3/Relu_grad/ReluGrad6train/gradients/layer_3/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_3/add_grad/ReshapeReshape$train/gradients/layer_3/add_grad/Sum&train/gradients/layer_3/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_3/add_grad/Sum_1Sum*train/gradients/layer_3/Relu_grad/ReluGrad8train/gradients/layer_3/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
*train/gradients/layer_3/add_grad/Reshape_1Reshape&train/gradients/layer_3/add_grad/Sum_1(train/gradients/layer_3/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_3/add_grad/tuple/group_depsNoOp)^train/gradients/layer_3/add_grad/Reshape+^train/gradients/layer_3/add_grad/Reshape_1
�
9train/gradients/layer_3/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_3/add_grad/Reshape2^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_3/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_3/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_3/add_grad/Reshape_12^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/weights3/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_2/Relu9train/gradients/layer_3/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_3/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_3/MatMul_grad/MatMul-^train/gradients/layer_3/MatMul_grad/MatMul_1
�
<train/gradients/layer_3/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_3/MatMul_grad/MatMul5^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_3/MatMul_grad/MatMul_15^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_3/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_2/Relu_grad/ReluGradReluGrad<train/gradients/layer_3/MatMul_grad/tuple/control_dependencylayer_2/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_2/add_grad/ShapeShapelayer_2/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_2/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_2/add_grad/Shape(train/gradients/layer_2/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_2/add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_2/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_2/add_grad/ReshapeReshape$train/gradients/layer_2/add_grad/Sum&train/gradients/layer_2/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_2/add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_2/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_2/add_grad/Reshape_1Reshape&train/gradients/layer_2/add_grad/Sum_1(train/gradients/layer_2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_2/add_grad/tuple/group_depsNoOp)^train/gradients/layer_2/add_grad/Reshape+^train/gradients/layer_2/add_grad/Reshape_1
�
9train/gradients/layer_2/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_2/add_grad/Reshape2^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_2/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_2/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_2/add_grad/Reshape_12^train/gradients/layer_2/add_grad/tuple/group_deps*
_output_shapes
:*
T0*=
_class3
1/loc:@train/gradients/layer_2/add_grad/Reshape_1
�
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/add_grad/tuple/control_dependencylayer_2/weights2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_1/Relu9train/gradients/layer_2/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
4train/gradients/layer_2/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_2/MatMul_grad/MatMul-^train/gradients/layer_2/MatMul_grad/MatMul_1
�
<train/gradients/layer_2/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_2/MatMul_grad/MatMul5^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_2/MatMul_grad/MatMul_15^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_2/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_1/Relu_grad/ReluGradReluGrad<train/gradients/layer_2/MatMul_grad/tuple/control_dependencylayer_1/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_1/add_grad/ShapeShapelayer_1/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_1/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/add_grad/Shape(train/gradients/layer_1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_1/add_grad/SumSum*train/gradients/layer_1/Relu_grad/ReluGrad6train/gradients/layer_1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_1/add_grad/ReshapeReshape$train/gradients/layer_1/add_grad/Sum&train/gradients/layer_1/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
&train/gradients/layer_1/add_grad/Sum_1Sum*train/gradients/layer_1/Relu_grad/ReluGrad8train/gradients/layer_1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_1/add_grad/Reshape_1Reshape&train/gradients/layer_1/add_grad/Sum_1(train/gradients/layer_1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_1/add_grad/tuple/group_depsNoOp)^train/gradients/layer_1/add_grad/Reshape+^train/gradients/layer_1/add_grad/Reshape_1
�
9train/gradients/layer_1/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_1/add_grad/Reshape2^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_1/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_1/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/add_grad/Reshape_12^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/add_grad/tuple/control_dependencylayer_1/weights1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_1/MatMul_grad/MatMul_1MatMulinput/Placeholder9train/gradients/layer_1/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
4train/gradients/layer_1/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_1/MatMul_grad/MatMul-^train/gradients/layer_1/MatMul_grad/MatMul_1
�
<train/gradients/layer_1/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_1/MatMul_grad/MatMul5^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_1/MatMul_grad/MatMul_15^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_1/MatMul_grad/MatMul_1*
_output_shapes

:
�
train/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *"
_class
loc:@layer_1/biases1*
valueB
 *fff?
�
train/beta1_power
VariableV2*"
_class
loc:@layer_1/biases1*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking(
z
train/beta1_power/readIdentitytrain/beta1_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *"
_class
loc:@layer_1/biases1*
valueB
 *w�?
�
train/beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@layer_1/biases1*
	container 
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
z
train/beta2_power/readIdentitytrain/beta2_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
-train/layer_1/weights1/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_1/weights1*
valueB*    
�
train/layer_1/weights1/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:
�
"train/layer_1/weights1/Adam/AssignAssigntrain/layer_1/weights1/Adam-train/layer_1/weights1/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
 train/layer_1/weights1/Adam/readIdentitytrain/layer_1/weights1/Adam*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
/train/layer_1/weights1/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_1/weights1/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:
�
$train/layer_1/weights1/Adam_1/AssignAssigntrain/layer_1/weights1/Adam_1/train/layer_1/weights1/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
"train/layer_1/weights1/Adam_1/readIdentitytrain/layer_1/weights1/Adam_1*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
,train/layer_1/biases1/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_1/biases1*
valueB*    
�
train/layer_1/biases1/Adam
VariableV2*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:*
dtype0*
_output_shapes
:
�
!train/layer_1/biases1/Adam/AssignAssigntrain/layer_1/biases1/Adam,train/layer_1/biases1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
train/layer_1/biases1/Adam/readIdentitytrain/layer_1/biases1/Adam*
_output_shapes
:*
T0*"
_class
loc:@layer_1/biases1
�
.train/layer_1/biases1/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_1/biases1/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:
�
#train/layer_1/biases1/Adam_1/AssignAssigntrain/layer_1/biases1/Adam_1.train/layer_1/biases1/Adam_1/Initializer/zeros*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:*
use_locking(
�
!train/layer_1/biases1/Adam_1/readIdentitytrain/layer_1/biases1/Adam_1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
-train/layer_2/weights2/Adam/Initializer/zerosConst*#
_class
loc:@layer_2/weights2*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_2/weights2/Adam
VariableV2*#
_class
loc:@layer_2/weights2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
"train/layer_2/weights2/Adam/AssignAssigntrain/layer_2/weights2/Adam-train/layer_2/weights2/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
 train/layer_2/weights2/Adam/readIdentitytrain/layer_2/weights2/Adam*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
/train/layer_2/weights2/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_2/weights2*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_2/weights2/Adam_1
VariableV2*#
_class
loc:@layer_2/weights2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
$train/layer_2/weights2/Adam_1/AssignAssigntrain/layer_2/weights2/Adam_1/train/layer_2/weights2/Adam_1/Initializer/zeros*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
"train/layer_2/weights2/Adam_1/readIdentitytrain/layer_2/weights2/Adam_1*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
,train/layer_2/biases2/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_2/biases2*
valueB*    
�
train/layer_2/biases2/Adam
VariableV2*"
_class
loc:@layer_2/biases2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
!train/layer_2/biases2/Adam/AssignAssigntrain/layer_2/biases2/Adam,train/layer_2/biases2/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
�
train/layer_2/biases2/Adam/readIdentitytrain/layer_2/biases2/Adam*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
.train/layer_2/biases2/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_2/biases2/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2*
	container *
shape:
�
#train/layer_2/biases2/Adam_1/AssignAssigntrain/layer_2/biases2/Adam_1.train/layer_2/biases2/Adam_1/Initializer/zeros*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
�
!train/layer_2/biases2/Adam_1/readIdentitytrain/layer_2/biases2/Adam_1*
_output_shapes
:*
T0*"
_class
loc:@layer_2/biases2
�
-train/layer_3/weights3/Adam/Initializer/zerosConst*#
_class
loc:@layer_3/weights3*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_3/weights3/Adam
VariableV2*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:*
dtype0*
_output_shapes

:
�
"train/layer_3/weights3/Adam/AssignAssigntrain/layer_3/weights3/Adam-train/layer_3/weights3/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
 train/layer_3/weights3/Adam/readIdentitytrain/layer_3/weights3/Adam*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
/train/layer_3/weights3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_3/weights3*
valueB*    
�
train/layer_3/weights3/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:
�
$train/layer_3/weights3/Adam_1/AssignAssigntrain/layer_3/weights3/Adam_1/train/layer_3/weights3/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
"train/layer_3/weights3/Adam_1/readIdentitytrain/layer_3/weights3/Adam_1*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
,train/layer_3/biases3/Adam/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_3/biases3/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:
�
!train/layer_3/biases3/Adam/AssignAssigntrain/layer_3/biases3/Adam,train/layer_3/biases3/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
train/layer_3/biases3/Adam/readIdentitytrain/layer_3/biases3/Adam*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
.train/layer_3/biases3/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_3/biases3/Adam_1
VariableV2*"
_class
loc:@layer_3/biases3*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
#train/layer_3/biases3/Adam_1/AssignAssigntrain/layer_3/biases3/Adam_1.train/layer_3/biases3/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
!train/layer_3/biases3/Adam_1/readIdentitytrain/layer_3/biases3/Adam_1*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
,train/output/weights4/Adam/Initializer/zerosConst*"
_class
loc:@output/weights4*
valueB*    *
dtype0*
_output_shapes

:
�
train/output/weights4/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4*
	container 
�
!train/output/weights4/Adam/AssignAssigntrain/output/weights4/Adam,train/output/weights4/Adam/Initializer/zeros*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
�
train/output/weights4/Adam/readIdentitytrain/output/weights4/Adam*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
.train/output/weights4/Adam_1/Initializer/zerosConst*"
_class
loc:@output/weights4*
valueB*    *
dtype0*
_output_shapes

:
�
train/output/weights4/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4*
	container *
shape
:
�
#train/output/weights4/Adam_1/AssignAssigntrain/output/weights4/Adam_1.train/output/weights4/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
�
!train/output/weights4/Adam_1/readIdentitytrain/output/weights4/Adam_1*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
+train/output/biases4/Adam/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
train/output/biases4/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@output/biases4*
	container 
�
 train/output/biases4/Adam/AssignAssigntrain/output/biases4/Adam+train/output/biases4/Adam/Initializer/zeros*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
�
train/output/biases4/Adam/readIdentitytrain/output/biases4/Adam*
_output_shapes
:*
T0*!
_class
loc:@output/biases4
�
-train/output/biases4/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*!
_class
loc:@output/biases4*
valueB*    
�
train/output/biases4/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@output/biases4
�
"train/output/biases4/Adam_1/AssignAssigntrain/output/biases4/Adam_1-train/output/biases4/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
�
 train/output/biases4/Adam_1/readIdentitytrain/output/biases4/Adam_1*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
]
train/Adam/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
U
train/Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
U
train/Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
W
train/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
,train/Adam/update_layer_1/weights1/ApplyAdam	ApplyAdamlayer_1/weights1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*#
_class
loc:@layer_1/weights1
�
+train/Adam/update_layer_1/biases1/ApplyAdam	ApplyAdamlayer_1/biases1train/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@layer_1/biases1
�
,train/Adam/update_layer_2/weights2/ApplyAdam	ApplyAdamlayer_2/weights2train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_2/weights2*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_2/biases2/ApplyAdam	ApplyAdamlayer_2/biases2train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@layer_2/biases2
�
,train/Adam/update_layer_3/weights3/ApplyAdam	ApplyAdamlayer_3/weights3train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_3/weights3*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_3/biases3/ApplyAdam	ApplyAdamlayer_3/biases3train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@layer_3/biases3
�
+train/Adam/update_output/weights4/ApplyAdam	ApplyAdamoutput/weights4train/output/weights4/Adamtrain/output/weights4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/output/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@output/weights4*
use_nesterov( *
_output_shapes

:
�
*train/Adam/update_output/biases4/ApplyAdam	ApplyAdamoutput/biases4train/output/biases4/Adamtrain/output/biases4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/output/add_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@output/biases4*
use_nesterov( *
_output_shapes
:
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking( 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking( 
�

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam
n
logging/current_cost/tagsConst*
dtype0*
_output_shapes
: *%
valueB Blogging/current_cost
l
logging/current_costScalarSummarylogging/current_cost/tags	cost/Mean*
_output_shapes
: *
T0
a
logging/Merge/MergeSummaryMergeSummarylogging/current_cost*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1*
dtype0*
_output_shapes
:
�
save/SaveV2/shape_and_slicesConst*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_sliceslayer_1/biases1layer_1/weights1layer_2/biases2layer_2/weights2layer_3/biases3layer_3/weights3output/biases4output/weights4train/beta1_powertrain/beta2_powertrain/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/output/biases4/Adamtrain/output/biases4/Adam_1train/output/weights4/Adamtrain/output/weights4/Adam_1*(
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*(
dtypes
2*|
_output_shapesj
h::::::::::::::::::::::::::
�
save/AssignAssignlayer_1/biases1save/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_1Assignlayer_1/weights1save/RestoreV2:1*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_2Assignlayer_2/biases2save/RestoreV2:2*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
save/Assign_3Assignlayer_2/weights2save/RestoreV2:3*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
save/Assign_4Assignlayer_3/biases3save/RestoreV2:4*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_5Assignlayer_3/weights3save/RestoreV2:5*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_6Assignoutput/biases4save/RestoreV2:6*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_7Assignoutput/weights4save/RestoreV2:7*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_8Assigntrain/beta1_powersave/RestoreV2:8*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
save/Assign_9Assigntrain/beta2_powersave/RestoreV2:9*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_10Assigntrain/layer_1/biases1/Adamsave/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_11Assigntrain/layer_1/biases1/Adam_1save/RestoreV2:11*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
save/Assign_12Assigntrain/layer_1/weights1/Adamsave/RestoreV2:12*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_13Assigntrain/layer_1/weights1/Adam_1save/RestoreV2:13*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_14Assigntrain/layer_2/biases2/Adamsave/RestoreV2:14*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
�
save/Assign_15Assigntrain/layer_2/biases2/Adam_1save/RestoreV2:15*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_16Assigntrain/layer_2/weights2/Adamsave/RestoreV2:16*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
save/Assign_17Assigntrain/layer_2/weights2/Adam_1save/RestoreV2:17*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_18Assigntrain/layer_3/biases3/Adamsave/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_19Assigntrain/layer_3/biases3/Adam_1save/RestoreV2:19*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_20Assigntrain/layer_3/weights3/Adamsave/RestoreV2:20*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
�
save/Assign_21Assigntrain/layer_3/weights3/Adam_1save/RestoreV2:21*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_22Assigntrain/output/biases4/Adamsave/RestoreV2:22*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_23Assigntrain/output/biases4/Adam_1save/RestoreV2:23*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_24Assigntrain/output/weights4/Adamsave/RestoreV2:24*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
�
save/Assign_25Assigntrain/output/weights4/Adam_1save/RestoreV2:25*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
initNoOp^layer_1/biases1/Assign^layer_1/weights1/Assign^layer_2/biases2/Assign^layer_2/weights2/Assign^layer_3/biases3/Assign^layer_3/weights3/Assign^output/biases4/Assign^output/weights4/Assign^train/beta1_power/Assign^train/beta2_power/Assign"^train/layer_1/biases1/Adam/Assign$^train/layer_1/biases1/Adam_1/Assign#^train/layer_1/weights1/Adam/Assign%^train/layer_1/weights1/Adam_1/Assign"^train/layer_2/biases2/Adam/Assign$^train/layer_2/biases2/Adam_1/Assign#^train/layer_2/weights2/Adam/Assign%^train/layer_2/weights2/Adam_1/Assign"^train/layer_3/biases3/Adam/Assign$^train/layer_3/biases3/Adam_1/Assign#^train/layer_3/weights3/Adam/Assign%^train/layer_3/weights3/Adam_1/Assign!^train/output/biases4/Adam/Assign#^train/output/biases4/Adam_1/Assign"^train/output/weights4/Adam/Assign$^train/output/weights4/Adam_1/Assign"�]��     ��d]	��?��AJ܉
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.12.02v1.12.0-0-ga6d8ffae09��
t
input/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
�
1layer_1/weights1/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_1/weights1*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_1/weights1/Initializer/random_uniform/minConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7��*
dtype0*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/maxConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7�?*
dtype0*
_output_shapes
: 
�
9layer_1/weights1/Initializer/random_uniform/RandomUniformRandomUniform1layer_1/weights1/Initializer/random_uniform/shape*
T0*#
_class
loc:@layer_1/weights1*
seed2 *
dtype0*
_output_shapes

:*

seed 
�
/layer_1/weights1/Initializer/random_uniform/subSub/layer_1/weights1/Initializer/random_uniform/max/layer_1/weights1/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@layer_1/weights1
�
/layer_1/weights1/Initializer/random_uniform/mulMul9layer_1/weights1/Initializer/random_uniform/RandomUniform/layer_1/weights1/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
+layer_1/weights1/Initializer/random_uniformAdd/layer_1/weights1/Initializer/random_uniform/mul/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
layer_1/weights1
VariableV2*#
_class
loc:@layer_1/weights1*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
layer_1/weights1/AssignAssignlayer_1/weights1+layer_1/weights1/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
�
layer_1/weights1/readIdentitylayer_1/weights1*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
�
!layer_1/biases1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
layer_1/biases1
VariableV2*"
_class
loc:@layer_1/biases1*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
layer_1/biases1/AssignAssignlayer_1/biases1!layer_1/biases1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
z
layer_1/biases1/readIdentitylayer_1/biases1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
layer_1/MatMulMatMulinput/Placeholderlayer_1/weights1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
j
layer_1/addAddlayer_1/MatMullayer_1/biases1/read*
T0*'
_output_shapes
:���������
S
layer_1/ReluRelulayer_1/add*
T0*'
_output_shapes
:���������
�
1layer_2/weights2/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_2/weights2*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_2/weights2/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *#
_class
loc:@layer_2/weights2*
valueB
 *׳]�
�
/layer_2/weights2/Initializer/random_uniform/maxConst*#
_class
loc:@layer_2/weights2*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
�
9layer_2/weights2/Initializer/random_uniform/RandomUniformRandomUniform1layer_2/weights2/Initializer/random_uniform/shape*
T0*#
_class
loc:@layer_2/weights2*
seed2 *
dtype0*
_output_shapes

:*

seed 
�
/layer_2/weights2/Initializer/random_uniform/subSub/layer_2/weights2/Initializer/random_uniform/max/layer_2/weights2/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@layer_2/weights2
�
/layer_2/weights2/Initializer/random_uniform/mulMul9layer_2/weights2/Initializer/random_uniform/RandomUniform/layer_2/weights2/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
+layer_2/weights2/Initializer/random_uniformAdd/layer_2/weights2/Initializer/random_uniform/mul/layer_2/weights2/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
layer_2/weights2
VariableV2*#
_class
loc:@layer_2/weights2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
layer_2/weights2/AssignAssignlayer_2/weights2+layer_2/weights2/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
layer_2/weights2/readIdentitylayer_2/weights2*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
!layer_2/biases2/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
layer_2/biases2
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2*
	container *
shape:
�
layer_2/biases2/AssignAssignlayer_2/biases2!layer_2/biases2/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
z
layer_2/biases2/readIdentitylayer_2/biases2*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
layer_2/MatMulMatMullayer_1/Relulayer_2/weights2/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
j
layer_2/addAddlayer_2/MatMullayer_2/biases2/read*
T0*'
_output_shapes
:���������
S
layer_2/ReluRelulayer_2/add*
T0*'
_output_shapes
:���������
�
1layer_3/weights3/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_3/weights3*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_3/weights3/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *#
_class
loc:@layer_3/weights3*
valueB
 *׳]�
�
/layer_3/weights3/Initializer/random_uniform/maxConst*#
_class
loc:@layer_3/weights3*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
�
9layer_3/weights3/Initializer/random_uniform/RandomUniformRandomUniform1layer_3/weights3/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_3/weights3*
seed2 
�
/layer_3/weights3/Initializer/random_uniform/subSub/layer_3/weights3/Initializer/random_uniform/max/layer_3/weights3/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes
: 
�
/layer_3/weights3/Initializer/random_uniform/mulMul9layer_3/weights3/Initializer/random_uniform/RandomUniform/layer_3/weights3/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
+layer_3/weights3/Initializer/random_uniformAdd/layer_3/weights3/Initializer/random_uniform/mul/layer_3/weights3/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
layer_3/weights3
VariableV2*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:*
dtype0*
_output_shapes

:
�
layer_3/weights3/AssignAssignlayer_3/weights3+layer_3/weights3/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
layer_3/weights3/readIdentitylayer_3/weights3*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
!layer_3/biases3/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
layer_3/biases3
VariableV2*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:*
dtype0*
_output_shapes
:
�
layer_3/biases3/AssignAssignlayer_3/biases3!layer_3/biases3/Initializer/zeros*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
z
layer_3/biases3/readIdentitylayer_3/biases3*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
layer_3/MatMulMatMullayer_2/Relulayer_3/weights3/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
j
layer_3/addAddlayer_3/MatMullayer_3/biases3/read*
T0*'
_output_shapes
:���������
S
layer_3/ReluRelulayer_3/add*
T0*'
_output_shapes
:���������
�
0output/weights4/Initializer/random_uniform/shapeConst*"
_class
loc:@output/weights4*
valueB"      *
dtype0*
_output_shapes
:
�
.output/weights4/Initializer/random_uniform/minConst*"
_class
loc:@output/weights4*
valueB
 *qĜ�*
dtype0*
_output_shapes
: 
�
.output/weights4/Initializer/random_uniform/maxConst*"
_class
loc:@output/weights4*
valueB
 *qĜ?*
dtype0*
_output_shapes
: 
�
8output/weights4/Initializer/random_uniform/RandomUniformRandomUniform0output/weights4/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*"
_class
loc:@output/weights4*
seed2 
�
.output/weights4/Initializer/random_uniform/subSub.output/weights4/Initializer/random_uniform/max.output/weights4/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@output/weights4
�
.output/weights4/Initializer/random_uniform/mulMul8output/weights4/Initializer/random_uniform/RandomUniform.output/weights4/Initializer/random_uniform/sub*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
*output/weights4/Initializer/random_uniformAdd.output/weights4/Initializer/random_uniform/mul.output/weights4/Initializer/random_uniform/min*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
output/weights4
VariableV2*"
_class
loc:@output/weights4*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
output/weights4/AssignAssignoutput/weights4*output/weights4/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
~
output/weights4/readIdentityoutput/weights4*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
 output/biases4/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
output/biases4
VariableV2*!
_class
loc:@output/biases4*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
output/biases4/AssignAssignoutput/biases4 output/biases4/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
w
output/biases4/readIdentityoutput/biases4*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
�
output/MatMulMatMullayer_3/Reluoutput/weights4/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
g

output/addAddoutput/MatMuloutput/biases4/read*
T0*'
_output_shapes
:���������
s
cost/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
{
cost/SquaredDifferenceSquaredDifference
output/addcost/Placeholder*
T0*'
_output_shapes
:���������
[

cost/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
s
	cost/MeanMeancost/SquaredDifference
cost/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
X
train/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
^
train/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
}
,train/gradients/cost/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
&train/gradients/cost/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/cost/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
z
$train/gradients/cost/Mean_grad/ShapeShapecost/SquaredDifference*
_output_shapes
:*
T0*
out_type0
�
#train/gradients/cost/Mean_grad/TileTile&train/gradients/cost/Mean_grad/Reshape$train/gradients/cost/Mean_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
|
&train/gradients/cost/Mean_grad/Shape_1Shapecost/SquaredDifference*
_output_shapes
:*
T0*
out_type0
i
&train/gradients/cost/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
n
$train/gradients/cost/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
#train/gradients/cost/Mean_grad/ProdProd&train/gradients/cost/Mean_grad/Shape_1$train/gradients/cost/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
p
&train/gradients/cost/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
%train/gradients/cost/Mean_grad/Prod_1Prod&train/gradients/cost/Mean_grad/Shape_2&train/gradients/cost/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
j
(train/gradients/cost/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
&train/gradients/cost/Mean_grad/MaximumMaximum%train/gradients/cost/Mean_grad/Prod_1(train/gradients/cost/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
'train/gradients/cost/Mean_grad/floordivFloorDiv#train/gradients/cost/Mean_grad/Prod&train/gradients/cost/Mean_grad/Maximum*
_output_shapes
: *
T0
�
#train/gradients/cost/Mean_grad/CastCast'train/gradients/cost/Mean_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
�
&train/gradients/cost/Mean_grad/truedivRealDiv#train/gradients/cost/Mean_grad/Tile#train/gradients/cost/Mean_grad/Cast*'
_output_shapes
:���������*
T0
{
1train/gradients/cost/SquaredDifference_grad/ShapeShape
output/add*
T0*
out_type0*
_output_shapes
:
�
3train/gradients/cost/SquaredDifference_grad/Shape_1Shapecost/Placeholder*
_output_shapes
:*
T0*
out_type0
�
Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/cost/SquaredDifference_grad/Shape3train/gradients/cost/SquaredDifference_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2train/gradients/cost/SquaredDifference_grad/scalarConst'^train/gradients/cost/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
/train/gradients/cost/SquaredDifference_grad/mulMul2train/gradients/cost/SquaredDifference_grad/scalar&train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/subSub
output/addcost/Placeholder'^train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/mul_1Mul/train/gradients/cost/SquaredDifference_grad/mul/train/gradients/cost/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/SumSum1train/gradients/cost/SquaredDifference_grad/mul_1Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
3train/gradients/cost/SquaredDifference_grad/ReshapeReshape/train/gradients/cost/SquaredDifference_grad/Sum1train/gradients/cost/SquaredDifference_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
1train/gradients/cost/SquaredDifference_grad/Sum_1Sum1train/gradients/cost/SquaredDifference_grad/mul_1Ctrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
5train/gradients/cost/SquaredDifference_grad/Reshape_1Reshape1train/gradients/cost/SquaredDifference_grad/Sum_13train/gradients/cost/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/NegNeg5train/gradients/cost/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
<train/gradients/cost/SquaredDifference_grad/tuple/group_depsNoOp0^train/gradients/cost/SquaredDifference_grad/Neg4^train/gradients/cost/SquaredDifference_grad/Reshape
�
Dtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependencyIdentity3train/gradients/cost/SquaredDifference_grad/Reshape=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/cost/SquaredDifference_grad/Reshape*'
_output_shapes
:���������
�
Ftrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/cost/SquaredDifference_grad/Neg=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/cost/SquaredDifference_grad/Neg*'
_output_shapes
:���������
r
%train/gradients/output/add_grad/ShapeShapeoutput/MatMul*
T0*
out_type0*
_output_shapes
:
q
'train/gradients/output/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
5train/gradients/output/add_grad/BroadcastGradientArgsBroadcastGradientArgs%train/gradients/output/add_grad/Shape'train/gradients/output/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
#train/gradients/output/add_grad/SumSumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency5train/gradients/output/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
'train/gradients/output/add_grad/ReshapeReshape#train/gradients/output/add_grad/Sum%train/gradients/output/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
%train/gradients/output/add_grad/Sum_1SumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency7train/gradients/output/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
)train/gradients/output/add_grad/Reshape_1Reshape%train/gradients/output/add_grad/Sum_1'train/gradients/output/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
0train/gradients/output/add_grad/tuple/group_depsNoOp(^train/gradients/output/add_grad/Reshape*^train/gradients/output/add_grad/Reshape_1
�
8train/gradients/output/add_grad/tuple/control_dependencyIdentity'train/gradients/output/add_grad/Reshape1^train/gradients/output/add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/output/add_grad/Reshape*'
_output_shapes
:���������
�
:train/gradients/output/add_grad/tuple/control_dependency_1Identity)train/gradients/output/add_grad/Reshape_11^train/gradients/output/add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/add_grad/Reshape_1*
_output_shapes
:
�
)train/gradients/output/MatMul_grad/MatMulMatMul8train/gradients/output/add_grad/tuple/control_dependencyoutput/weights4/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
+train/gradients/output/MatMul_grad/MatMul_1MatMullayer_3/Relu8train/gradients/output/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
3train/gradients/output/MatMul_grad/tuple/group_depsNoOp*^train/gradients/output/MatMul_grad/MatMul,^train/gradients/output/MatMul_grad/MatMul_1
�
;train/gradients/output/MatMul_grad/tuple/control_dependencyIdentity)train/gradients/output/MatMul_grad/MatMul4^train/gradients/output/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*<
_class2
0.loc:@train/gradients/output/MatMul_grad/MatMul
�
=train/gradients/output/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/output/MatMul_grad/MatMul_14^train/gradients/output/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/output/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_3/Relu_grad/ReluGradReluGrad;train/gradients/output/MatMul_grad/tuple/control_dependencylayer_3/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_3/add_grad/ShapeShapelayer_3/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_3/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/add_grad/Shape(train/gradients/layer_3/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$train/gradients/layer_3/add_grad/SumSum*train/gradients/layer_3/Relu_grad/ReluGrad6train/gradients/layer_3/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
(train/gradients/layer_3/add_grad/ReshapeReshape$train/gradients/layer_3/add_grad/Sum&train/gradients/layer_3/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_3/add_grad/Sum_1Sum*train/gradients/layer_3/Relu_grad/ReluGrad8train/gradients/layer_3/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
*train/gradients/layer_3/add_grad/Reshape_1Reshape&train/gradients/layer_3/add_grad/Sum_1(train/gradients/layer_3/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_3/add_grad/tuple/group_depsNoOp)^train/gradients/layer_3/add_grad/Reshape+^train/gradients/layer_3/add_grad/Reshape_1
�
9train/gradients/layer_3/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_3/add_grad/Reshape2^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_3/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_3/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_3/add_grad/Reshape_12^train/gradients/layer_3/add_grad/tuple/group_deps*
_output_shapes
:*
T0*=
_class3
1/loc:@train/gradients/layer_3/add_grad/Reshape_1
�
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/weights3/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_2/Relu9train/gradients/layer_3/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_3/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_3/MatMul_grad/MatMul-^train/gradients/layer_3/MatMul_grad/MatMul_1
�
<train/gradients/layer_3/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_3/MatMul_grad/MatMul5^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_3/MatMul_grad/MatMul_15^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_3/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_2/Relu_grad/ReluGradReluGrad<train/gradients/layer_3/MatMul_grad/tuple/control_dependencylayer_2/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_2/add_grad/ShapeShapelayer_2/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_2/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_2/add_grad/Shape(train/gradients/layer_2/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_2/add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_2/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_2/add_grad/ReshapeReshape$train/gradients/layer_2/add_grad/Sum&train/gradients/layer_2/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_2/add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_2/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
*train/gradients/layer_2/add_grad/Reshape_1Reshape&train/gradients/layer_2/add_grad/Sum_1(train/gradients/layer_2/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
1train/gradients/layer_2/add_grad/tuple/group_depsNoOp)^train/gradients/layer_2/add_grad/Reshape+^train/gradients/layer_2/add_grad/Reshape_1
�
9train/gradients/layer_2/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_2/add_grad/Reshape2^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_2/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_2/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_2/add_grad/Reshape_12^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/add_grad/tuple/control_dependencylayer_2/weights2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_1/Relu9train/gradients/layer_2/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_2/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_2/MatMul_grad/MatMul-^train/gradients/layer_2/MatMul_grad/MatMul_1
�
<train/gradients/layer_2/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_2/MatMul_grad/MatMul5^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_2/MatMul_grad/MatMul_15^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_2/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_1/Relu_grad/ReluGradReluGrad<train/gradients/layer_2/MatMul_grad/tuple/control_dependencylayer_1/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_1/add_grad/ShapeShapelayer_1/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_1/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/add_grad/Shape(train/gradients/layer_1/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$train/gradients/layer_1/add_grad/SumSum*train/gradients/layer_1/Relu_grad/ReluGrad6train/gradients/layer_1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
(train/gradients/layer_1/add_grad/ReshapeReshape$train/gradients/layer_1/add_grad/Sum&train/gradients/layer_1/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
&train/gradients/layer_1/add_grad/Sum_1Sum*train/gradients/layer_1/Relu_grad/ReluGrad8train/gradients/layer_1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_1/add_grad/Reshape_1Reshape&train/gradients/layer_1/add_grad/Sum_1(train/gradients/layer_1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_1/add_grad/tuple/group_depsNoOp)^train/gradients/layer_1/add_grad/Reshape+^train/gradients/layer_1/add_grad/Reshape_1
�
9train/gradients/layer_1/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_1/add_grad/Reshape2^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_1/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_1/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/add_grad/Reshape_12^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/add_grad/tuple/control_dependencylayer_1/weights1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_1/MatMul_grad/MatMul_1MatMulinput/Placeholder9train/gradients/layer_1/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
4train/gradients/layer_1/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_1/MatMul_grad/MatMul-^train/gradients/layer_1/MatMul_grad/MatMul_1
�
<train/gradients/layer_1/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_1/MatMul_grad/MatMul5^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_1/MatMul_grad/MatMul_15^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*?
_class5
31loc:@train/gradients/layer_1/MatMul_grad/MatMul_1
�
train/beta1_power/initial_valueConst*"
_class
loc:@layer_1/biases1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
train/beta1_power
VariableV2*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape: *
dtype0*
_output_shapes
: 
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
z
train/beta1_power/readIdentitytrain/beta1_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/beta2_power/initial_valueConst*"
_class
loc:@layer_1/biases1*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
train/beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@layer_1/biases1
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking(
z
train/beta2_power/readIdentitytrain/beta2_power*
_output_shapes
: *
T0*"
_class
loc:@layer_1/biases1
�
-train/layer_1/weights1/Adam/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_1/weights1/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:
�
"train/layer_1/weights1/Adam/AssignAssigntrain/layer_1/weights1/Adam-train/layer_1/weights1/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
 train/layer_1/weights1/Adam/readIdentitytrain/layer_1/weights1/Adam*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
�
/train/layer_1/weights1/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_1/weights1/Adam_1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1*
	container 
�
$train/layer_1/weights1/Adam_1/AssignAssigntrain/layer_1/weights1/Adam_1/train/layer_1/weights1/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
"train/layer_1/weights1/Adam_1/readIdentitytrain/layer_1/weights1/Adam_1*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
,train/layer_1/biases1/Adam/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_1/biases1/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:
�
!train/layer_1/biases1/Adam/AssignAssigntrain/layer_1/biases1/Adam,train/layer_1/biases1/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
train/layer_1/biases1/Adam/readIdentitytrain/layer_1/biases1/Adam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
.train/layer_1/biases1/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_1/biases1/Adam_1
VariableV2*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:*
dtype0*
_output_shapes
:
�
#train/layer_1/biases1/Adam_1/AssignAssigntrain/layer_1/biases1/Adam_1.train/layer_1/biases1/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
!train/layer_1/biases1/Adam_1/readIdentitytrain/layer_1/biases1/Adam_1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
-train/layer_2/weights2/Adam/Initializer/zerosConst*#
_class
loc:@layer_2/weights2*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_2/weights2/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:
�
"train/layer_2/weights2/Adam/AssignAssigntrain/layer_2/weights2/Adam-train/layer_2/weights2/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
�
 train/layer_2/weights2/Adam/readIdentitytrain/layer_2/weights2/Adam*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
/train/layer_2/weights2/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_2/weights2*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_2/weights2/Adam_1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2*
	container 
�
$train/layer_2/weights2/Adam_1/AssignAssigntrain/layer_2/weights2/Adam_1/train/layer_2/weights2/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
"train/layer_2/weights2/Adam_1/readIdentitytrain/layer_2/weights2/Adam_1*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
,train/layer_2/biases2/Adam/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_2/biases2/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2
�
!train/layer_2/biases2/Adam/AssignAssigntrain/layer_2/biases2/Adam,train/layer_2/biases2/Adam/Initializer/zeros*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
�
train/layer_2/biases2/Adam/readIdentitytrain/layer_2/biases2/Adam*
_output_shapes
:*
T0*"
_class
loc:@layer_2/biases2
�
.train/layer_2/biases2/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_2/biases2*
valueB*    
�
train/layer_2/biases2/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2
�
#train/layer_2/biases2/Adam_1/AssignAssigntrain/layer_2/biases2/Adam_1.train/layer_2/biases2/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
!train/layer_2/biases2/Adam_1/readIdentitytrain/layer_2/biases2/Adam_1*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
-train/layer_3/weights3/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_3/weights3*
valueB*    
�
train/layer_3/weights3/Adam
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3
�
"train/layer_3/weights3/Adam/AssignAssigntrain/layer_3/weights3/Adam-train/layer_3/weights3/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
 train/layer_3/weights3/Adam/readIdentitytrain/layer_3/weights3/Adam*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
/train/layer_3/weights3/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_3/weights3*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_3/weights3/Adam_1
VariableV2*#
_class
loc:@layer_3/weights3*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
$train/layer_3/weights3/Adam_1/AssignAssigntrain/layer_3/weights3/Adam_1/train/layer_3/weights3/Adam_1/Initializer/zeros*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
"train/layer_3/weights3/Adam_1/readIdentitytrain/layer_3/weights3/Adam_1*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
,train/layer_3/biases3/Adam/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_3/biases3/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:
�
!train/layer_3/biases3/Adam/AssignAssigntrain/layer_3/biases3/Adam,train/layer_3/biases3/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
train/layer_3/biases3/Adam/readIdentitytrain/layer_3/biases3/Adam*
_output_shapes
:*
T0*"
_class
loc:@layer_3/biases3
�
.train/layer_3/biases3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_3/biases3*
valueB*    
�
train/layer_3/biases3/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:
�
#train/layer_3/biases3/Adam_1/AssignAssigntrain/layer_3/biases3/Adam_1.train/layer_3/biases3/Adam_1/Initializer/zeros*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
�
!train/layer_3/biases3/Adam_1/readIdentitytrain/layer_3/biases3/Adam_1*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
,train/output/weights4/Adam/Initializer/zerosConst*"
_class
loc:@output/weights4*
valueB*    *
dtype0*
_output_shapes

:
�
train/output/weights4/Adam
VariableV2*"
_class
loc:@output/weights4*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
!train/output/weights4/Adam/AssignAssigntrain/output/weights4/Adam,train/output/weights4/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
�
train/output/weights4/Adam/readIdentitytrain/output/weights4/Adam*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
.train/output/weights4/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*"
_class
loc:@output/weights4*
valueB*    
�
train/output/weights4/Adam_1
VariableV2*
shared_name *"
_class
loc:@output/weights4*
	container *
shape
:*
dtype0*
_output_shapes

:
�
#train/output/weights4/Adam_1/AssignAssigntrain/output/weights4/Adam_1.train/output/weights4/Adam_1/Initializer/zeros*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
�
!train/output/weights4/Adam_1/readIdentitytrain/output/weights4/Adam_1*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
+train/output/biases4/Adam/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
train/output/biases4/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:
�
 train/output/biases4/Adam/AssignAssigntrain/output/biases4/Adam+train/output/biases4/Adam/Initializer/zeros*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
�
train/output/biases4/Adam/readIdentitytrain/output/biases4/Adam*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
�
-train/output/biases4/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*!
_class
loc:@output/biases4*
valueB*    
�
train/output/biases4/Adam_1
VariableV2*!
_class
loc:@output/biases4*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
"train/output/biases4/Adam_1/AssignAssigntrain/output/biases4/Adam_1-train/output/biases4/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
�
 train/output/biases4/Adam_1/readIdentitytrain/output/biases4/Adam_1*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
]
train/Adam/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
U
train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
,train/Adam/update_layer_1/weights1/ApplyAdam	ApplyAdamlayer_1/weights1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_1/weights1*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_1/biases1/ApplyAdam	ApplyAdamlayer_1/biases1train/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/add_grad/tuple/control_dependency_1*
T0*"
_class
loc:@layer_1/biases1*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
,train/Adam/update_layer_2/weights2/ApplyAdam	ApplyAdamlayer_2/weights2train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_2/weights2*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_2/biases2/ApplyAdam	ApplyAdamlayer_2/biases2train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_2/biases2*
use_nesterov( *
_output_shapes
:
�
,train/Adam/update_layer_3/weights3/ApplyAdam	ApplyAdamlayer_3/weights3train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1*
T0*#
_class
loc:@layer_3/weights3*
use_nesterov( *
_output_shapes

:*
use_locking( 
�
+train/Adam/update_layer_3/biases3/ApplyAdam	ApplyAdamlayer_3/biases3train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/add_grad/tuple/control_dependency_1*
T0*"
_class
loc:@layer_3/biases3*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
+train/Adam/update_output/weights4/ApplyAdam	ApplyAdamoutput/weights4train/output/weights4/Adamtrain/output/weights4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/output/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@output/weights4*
use_nesterov( *
_output_shapes

:
�
*train/Adam/update_output/biases4/ApplyAdam	ApplyAdamoutput/biases4train/output/biases4/Adamtrain/output/biases4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/output/add_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@output/biases4*
use_nesterov( *
_output_shapes
:
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
use_locking( *
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
use_locking( *
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
�

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam
n
logging/current_cost/tagsConst*%
valueB Blogging/current_cost*
dtype0*
_output_shapes
: 
l
logging/current_costScalarSummarylogging/current_cost/tags	cost/Mean*
T0*
_output_shapes
: 
a
logging/Merge/MergeSummaryMergeSummarylogging/current_cost*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1
�
save/SaveV2/shape_and_slicesConst*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_sliceslayer_1/biases1layer_1/weights1layer_2/biases2layer_2/weights2layer_3/biases3layer_3/weights3output/biases4output/weights4train/beta1_powertrain/beta2_powertrain/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/output/biases4/Adamtrain/output/biases4/Adam_1train/output/weights4/Adamtrain/output/weights4/Adam_1*(
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1*
dtype0*
_output_shapes
:
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2
�
save/AssignAssignlayer_1/biases1save/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_1Assignlayer_1/weights1save/RestoreV2:1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
�
save/Assign_2Assignlayer_2/biases2save/RestoreV2:2*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
save/Assign_3Assignlayer_2/weights2save/RestoreV2:3*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_4Assignlayer_3/biases3save/RestoreV2:4*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_5Assignlayer_3/weights3save/RestoreV2:5*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_6Assignoutput/biases4save/RestoreV2:6*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
�
save/Assign_7Assignoutput/weights4save/RestoreV2:7*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_8Assigntrain/beta1_powersave/RestoreV2:8*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
save/Assign_9Assigntrain/beta2_powersave/RestoreV2:9*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
save/Assign_10Assigntrain/layer_1/biases1/Adamsave/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_11Assigntrain/layer_1/biases1/Adam_1save/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_12Assigntrain/layer_1/weights1/Adamsave/RestoreV2:12*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_13Assigntrain/layer_1/weights1/Adam_1save/RestoreV2:13*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_14Assigntrain/layer_2/biases2/Adamsave/RestoreV2:14*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
save/Assign_15Assigntrain/layer_2/biases2/Adam_1save/RestoreV2:15*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_16Assigntrain/layer_2/weights2/Adamsave/RestoreV2:16*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
�
save/Assign_17Assigntrain/layer_2/weights2/Adam_1save/RestoreV2:17*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_18Assigntrain/layer_3/biases3/Adamsave/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_19Assigntrain/layer_3/biases3/Adam_1save/RestoreV2:19*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_20Assigntrain/layer_3/weights3/Adamsave/RestoreV2:20*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_21Assigntrain/layer_3/weights3/Adam_1save/RestoreV2:21*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_22Assigntrain/output/biases4/Adamsave/RestoreV2:22*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
�
save/Assign_23Assigntrain/output/biases4/Adam_1save/RestoreV2:23*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_24Assigntrain/output/weights4/Adamsave/RestoreV2:24*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_25Assigntrain/output/weights4/Adam_1save/RestoreV2:25*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
initNoOp^layer_1/biases1/Assign^layer_1/weights1/Assign^layer_2/biases2/Assign^layer_2/weights2/Assign^layer_3/biases3/Assign^layer_3/weights3/Assign^output/biases4/Assign^output/weights4/Assign^train/beta1_power/Assign^train/beta2_power/Assign"^train/layer_1/biases1/Adam/Assign$^train/layer_1/biases1/Adam_1/Assign#^train/layer_1/weights1/Adam/Assign%^train/layer_1/weights1/Adam_1/Assign"^train/layer_2/biases2/Adam/Assign$^train/layer_2/biases2/Adam_1/Assign#^train/layer_2/weights2/Adam/Assign%^train/layer_2/weights2/Adam_1/Assign"^train/layer_3/biases3/Adam/Assign$^train/layer_3/biases3/Adam_1/Assign#^train/layer_3/weights3/Adam/Assign%^train/layer_3/weights3/Adam_1/Assign!^train/output/biases4/Adam/Assign#^train/output/biases4/Adam_1/Assign"^train/output/weights4/Adam/Assign$^train/output/weights4/Adam_1/Assign""'
	summaries

logging/current_cost:0"�
trainable_variables��
w
layer_1/weights1:0layer_1/weights1/Assignlayer_1/weights1/read:02-layer_1/weights1/Initializer/random_uniform:08
j
layer_1/biases1:0layer_1/biases1/Assignlayer_1/biases1/read:02#layer_1/biases1/Initializer/zeros:08
w
layer_2/weights2:0layer_2/weights2/Assignlayer_2/weights2/read:02-layer_2/weights2/Initializer/random_uniform:08
j
layer_2/biases2:0layer_2/biases2/Assignlayer_2/biases2/read:02#layer_2/biases2/Initializer/zeros:08
w
layer_3/weights3:0layer_3/weights3/Assignlayer_3/weights3/read:02-layer_3/weights3/Initializer/random_uniform:08
j
layer_3/biases3:0layer_3/biases3/Assignlayer_3/biases3/read:02#layer_3/biases3/Initializer/zeros:08
s
output/weights4:0output/weights4/Assignoutput/weights4/read:02,output/weights4/Initializer/random_uniform:08
f
output/biases4:0output/biases4/Assignoutput/biases4/read:02"output/biases4/Initializer/zeros:08"
train_op


train/Adam"�
	variables��
w
layer_1/weights1:0layer_1/weights1/Assignlayer_1/weights1/read:02-layer_1/weights1/Initializer/random_uniform:08
j
layer_1/biases1:0layer_1/biases1/Assignlayer_1/biases1/read:02#layer_1/biases1/Initializer/zeros:08
w
layer_2/weights2:0layer_2/weights2/Assignlayer_2/weights2/read:02-layer_2/weights2/Initializer/random_uniform:08
j
layer_2/biases2:0layer_2/biases2/Assignlayer_2/biases2/read:02#layer_2/biases2/Initializer/zeros:08
w
layer_3/weights3:0layer_3/weights3/Assignlayer_3/weights3/read:02-layer_3/weights3/Initializer/random_uniform:08
j
layer_3/biases3:0layer_3/biases3/Assignlayer_3/biases3/read:02#layer_3/biases3/Initializer/zeros:08
s
output/weights4:0output/weights4/Assignoutput/weights4/read:02,output/weights4/Initializer/random_uniform:08
f
output/biases4:0output/biases4/Assignoutput/biases4/read:02"output/biases4/Initializer/zeros:08
l
train/beta1_power:0train/beta1_power/Assigntrain/beta1_power/read:02!train/beta1_power/initial_value:0
l
train/beta2_power:0train/beta2_power/Assigntrain/beta2_power/read:02!train/beta2_power/initial_value:0
�
train/layer_1/weights1/Adam:0"train/layer_1/weights1/Adam/Assign"train/layer_1/weights1/Adam/read:02/train/layer_1/weights1/Adam/Initializer/zeros:0
�
train/layer_1/weights1/Adam_1:0$train/layer_1/weights1/Adam_1/Assign$train/layer_1/weights1/Adam_1/read:021train/layer_1/weights1/Adam_1/Initializer/zeros:0
�
train/layer_1/biases1/Adam:0!train/layer_1/biases1/Adam/Assign!train/layer_1/biases1/Adam/read:02.train/layer_1/biases1/Adam/Initializer/zeros:0
�
train/layer_1/biases1/Adam_1:0#train/layer_1/biases1/Adam_1/Assign#train/layer_1/biases1/Adam_1/read:020train/layer_1/biases1/Adam_1/Initializer/zeros:0
�
train/layer_2/weights2/Adam:0"train/layer_2/weights2/Adam/Assign"train/layer_2/weights2/Adam/read:02/train/layer_2/weights2/Adam/Initializer/zeros:0
�
train/layer_2/weights2/Adam_1:0$train/layer_2/weights2/Adam_1/Assign$train/layer_2/weights2/Adam_1/read:021train/layer_2/weights2/Adam_1/Initializer/zeros:0
�
train/layer_2/biases2/Adam:0!train/layer_2/biases2/Adam/Assign!train/layer_2/biases2/Adam/read:02.train/layer_2/biases2/Adam/Initializer/zeros:0
�
train/layer_2/biases2/Adam_1:0#train/layer_2/biases2/Adam_1/Assign#train/layer_2/biases2/Adam_1/read:020train/layer_2/biases2/Adam_1/Initializer/zeros:0
�
train/layer_3/weights3/Adam:0"train/layer_3/weights3/Adam/Assign"train/layer_3/weights3/Adam/read:02/train/layer_3/weights3/Adam/Initializer/zeros:0
�
train/layer_3/weights3/Adam_1:0$train/layer_3/weights3/Adam_1/Assign$train/layer_3/weights3/Adam_1/read:021train/layer_3/weights3/Adam_1/Initializer/zeros:0
�
train/layer_3/biases3/Adam:0!train/layer_3/biases3/Adam/Assign!train/layer_3/biases3/Adam/read:02.train/layer_3/biases3/Adam/Initializer/zeros:0
�
train/layer_3/biases3/Adam_1:0#train/layer_3/biases3/Adam_1/Assign#train/layer_3/biases3/Adam_1/read:020train/layer_3/biases3/Adam_1/Initializer/zeros:0
�
train/output/weights4/Adam:0!train/output/weights4/Adam/Assign!train/output/weights4/Adam/read:02.train/output/weights4/Adam/Initializer/zeros:0
�
train/output/weights4/Adam_1:0#train/output/weights4/Adam_1/Assign#train/output/weights4/Adam_1/read:020train/output/weights4/Adam_1/Initializer/zeros:0
�
train/output/biases4/Adam:0 train/output/biases4/Adam/Assign train/output/biases4/Adam/read:02-train/output/biases4/Adam/Initializer/zeros:0
�
train/output/biases4/Adam_1:0"train/output/biases4/Adam_1/Assign"train/output/biases4/Adam_1/read:02/train/output/biases4/Adam_1/Initializer/zeros:0�k-|(       �pJ	[�B��A*

logging/current_cost�)=(���*       ����	��B��A*

logging/current_cost܀�<��*       ����	a"C��A
*

logging/current_cost��<$m��*       ����	�UC��A*

logging/current_costO��<h٪�*       ����	s�C��A*

logging/current_cost�O�<'X�*       ����	�C��A*

logging/current_cost�Ŷ<_z�*       ����	�C��A*

logging/current_cost�<�	�*       ����	$D��A#*

logging/current_costW��<�wĕ*       ����	�ND��A(*

logging/current_costި<�Ey*       ����	�D��A-*

logging/current_cost�{�<��7l*       ����	E�D��A2*

logging/current_cost���<����*       ����	��D��A7*

logging/current_cost)H�<?�*       ����	HE��A<*

logging/current_cost���<��*       ����	E9E��AA*

logging/current_cost�N�<7��|*       ����	�uE��AF*

logging/current_cost���<��4�*       ����	2�E��AK*

logging/current_cost�у<�tU*       ����	&�E��AP*

logging/current_costx<��[Q*       ����	��E��AU*

logging/current_costdj<���J*       ����	�+F��AZ*

logging/current_costVi]<G�t+*       ����	�XF��A_*

logging/current_cost�AR<H��*       ����	X�F��Ad*

logging/current_cost��H<����*       ����	�F��Ai*

logging/current_cost��@<h"G�*       ����	��F��An*

logging/current_costG�8<@I��*       ����	�G��As*

logging/current_cost�1<��x�*       ����	N>G��Ax*

logging/current_costl�*<�#�*       ����	kG��A}*

logging/current_cost��$<ؠ��+       ��K	��G��A�*

logging/current_costl`<f��j+       ��K	��G��A�*

logging/current_cost�<��R�+       ��K	x�G��A�*

logging/current_cost�I<d)��+       ��K	U%H��A�*

logging/current_cost=�<w��+       ��K	�QH��A�*

logging/current_coste�<�L�p+       ��K	}H��A�*

logging/current_cost�<�X+       ��K	޲H��A�*

logging/current_cost�<6��+       ��K	�H��A�*

logging/current_costg'	<�a+�+       ��K	iI��A�*

logging/current_cost�<ն�S+       ��K	�9I��A�*

logging/current_coste�<j�7�+       ��K	:hI��A�*

logging/current_cost�W<yqR�+       ��K	p�I��A�*

logging/current_cost��<�dnh+       ��K	u�I��A�*

logging/current_costtm <��q+       ��K	��I��A�*

logging/current_cost���;g�;+       ��K	-J��A�*

logging/current_cost2o�;�ʕ�+       ��K	`[J��A�*

logging/current_costPV�;Ћi+       ��K	ևJ��A�*

logging/current_costu��;�p+       ��K	��J��A�*

logging/current_cost��;oX�B+       ��K	��J��A�*

logging/current_cost�j�;��jN+       ��K	nK��A�*

logging/current_cost���;5%+       ��K	ECK��A�*

logging/current_cost���;եY�+       ��K	�qK��A�*

logging/current_cost���;m�:b+       ��K	�K��A�*

logging/current_cost�B�;4��/+       ��K	��K��A�*

logging/current_costdT�;�/�+       ��K	M�K��A�*

logging/current_costE��;�)8�+       ��K	)L��A�*

logging/current_cost���;��+       ��K	:VL��A�*

logging/current_cost���;�LW+       ��K	 �L��A�*

logging/current_cost�S�;�^Rt+       ��K	ñL��A�*

logging/current_cost���;{am�+       ��K	d�L��A�*

logging/current_cost7V�;}�x+       ��K	�M��A�*

logging/current_cost�|�;C<�+       ��K		;M��A�*

logging/current_cost���;r��+       ��K	LhM��A�*

logging/current_cost���;�:�+       ��K	�M��A�*

logging/current_cost��;��׬+       ��K	�M��A�*

logging/current_cost���;�(?+       ��K	��M��A�*

logging/current_cost��;ۄ1e+       ��K	�7N��A�*

logging/current_cost~{�;Ծ��+       ��K	xsN��A�*

logging/current_cost�m�;��S+       ��K	�N��A�*

logging/current_cost	��;&ads+       ��K	��N��A�*

logging/current_cost��;��Vq+       ��K	�/O��A�*

logging/current_cost)��;zg�^+       ��K	�~O��A�*

logging/current_cost{F�;�<�+       ��K	�O��A�*

logging/current_cost'i�;�V+       ��K	�P��A�*

logging/current_cost��;x�!k+       ��K	�?P��A�*

logging/current_cost���; b�+       ��K	nuP��A�*

logging/current_cost	;�;���y+       ��K	��P��A�*

logging/current_cost^��;��M+       ��K	��P��A�*

logging/current_cost�'�;I=�+       ��K	0Q��A�*

logging/current_costn�;��\Q+       ��K	�RQ��A�*

logging/current_cost��;Z��)+       ��K	y�Q��A�*

logging/current_cost��;)��+       ��K	�Q��A�*

logging/current_cost��;d�+       ��K	��Q��A�*

logging/current_costN2�;�v'�+       ��K	R��A�*

logging/current_costY��;����+       ��K	NTR��A�*

logging/current_cost�-�;����+       ��K	��R��A�*

logging/current_cost��;�r� +       ��K	�R��A�*

logging/current_cost���;���l+       ��K	F�R��A�*

logging/current_costB��;.�Y�+       ��K	�S��A�*

logging/current_cost�?�;��F+       ��K	LJS��A�*

logging/current_costG.�;��E�+       ��K	 ~S��A�*

logging/current_cost>��;��ٕ+       ��K	�S��A�*

logging/current_cost��;Jgk�+       ��K	6�S��A�*

logging/current_cost�n�;�sԉ+       ��K	�T��A�*

logging/current_costTD�;g��++       ��K	�NT��A�*

logging/current_costr3�;?4�+       ��K	��T��A�*

logging/current_cost.'�;Ip�+       ��K	 �T��A�*

logging/current_cost�I�;��t�+       ��K	��T��A�*

logging/current_cost�O�;Ֆ�+       ��K	%U��A�*

logging/current_costrf�;��X�+       ��K	`QU��A�*

logging/current_cost�V�;�d��+       ��K	��U��A�*

logging/current_cost~�;�J�c+       ��K	߽U��A�*

logging/current_costRk�;�b�+       ��K	��U��A�*

logging/current_costՍ�;���E+       ��K	}V��A�*

logging/current_costN��;(E+       ��K	�NV��A�*

logging/current_cost��;���e+       ��K	�}V��A�*

logging/current_cost5��;D��+       ��K	�V��A�*

logging/current_cost$��;�4�^+       ��K	r�V��A�*

logging/current_cost���;qn�+       ��K	�W��A�*

logging/current_cost���;l�\�+       ��K	a5W��A�*

logging/current_costl��;+�I]+       ��K	�aW��A�*

logging/current_cost���;"�,+       ��K	��W��A�*

logging/current_cost���;I��+       ��K	�W��A�*

logging/current_cost��;b��+       ��K	�W��A�*

logging/current_costɻ�;}x'�+       ��K	�X��A�*

logging/current_cost���;<;�+       ��K	�KX��A�*

logging/current_cost���;1�i+       ��K	exX��A�*

logging/current_cost���;��7+       ��K	�X��A�*

logging/current_cost\��;�D�+       ��K	k�X��A�*

logging/current_cost'��;I�Z+       ��K	�Y��A�*

logging/current_cost���;QS�C+       ��K	20Y��A�*

logging/current_costr��;IM�v+       ��K	�\Y��A�*

logging/current_costK��;=��+       ��K	�Y��A�*

logging/current_costL��;��mW+       ��K	��Y��A�*

logging/current_cost���;���+       ��K	�Z��A�*

logging/current_costG�;�W�+       ��K	�4Z��A�*

logging/current_cost�;�kX�+       ��K	�`Z��A�*

logging/current_cost'�;�2�+       ��K	�Z��A�*

logging/current_cost�6�;��7�+       ��K	2�Z��A�*

logging/current_cost�3�;؀�U+       ��K	��Z��A�*

logging/current_costdA�;u�M@+       ��K	'"[��A�*

logging/current_costuH�;� +       ��K	�N[��A�*

logging/current_costZ�;����+       ��K	�{[��A�*

logging/current_cost�o�;���=+       ��K	��[��A�*

logging/current_coste��;�T�2+       ��K	��[��A�*

logging/current_cost���;߆�+       ��K	d\��A�*

logging/current_cost��;�㊡+       ��K	j/\��A�*

logging/current_cost˳�;���b+       ��K	+]\��A�*

logging/current_costD��;�N�c+       ��K	O�\��A�*

logging/current_costT��;��bP+       ��K	Ǻ\��A�*

logging/current_costT��;��8�+       ��K	��\��A�*

logging/current_cost��;h���+       ��K	)]��A�*

logging/current_costg[�;Q�6�+       ��K	�T]��A�*

logging/current_cost~I�;����+       ��K	y�]��A�*

logging/current_costp.�;��)+       ��K	�]��A�*

logging/current_cost�=�;s��+       ��K	��]��A�*

logging/current_cost�R�;+|��+       ��K	�^��A�*

logging/current_cost�G�;�?J+       ��K	�?^��A�*

logging/current_cost�9�;�>!+       ��K	<s^��A�*

logging/current_costO�;Ù+       ��K	��^��A�*

logging/current_costdX�;���+       ��K	��^��A�*

logging/current_cost+S�;��Ν+       ��K	p�^��A�*

logging/current_cost�d�;Rb�+       ��K	D2_��A�*

logging/current_coste��;�6J+       ��K	Vc_��A�*

logging/current_costiO�;@�+       ��K	��_��A�*

logging/current_costІ�;��q9+       ��K	��_��A�*

logging/current_cost�x�;|S�+       ��K	��_��A�*

logging/current_costDy�;�X�*+       ��K	;%`��A�*

logging/current_cost̊�;b'�+       ��K	�T`��A�*

logging/current_costE�;�S�+       ��K	�`��A�*

logging/current_costd��;5�+       ��K	0�`��A�*

logging/current_costׁ�;C�+       ��K	0�`��A�*

logging/current_cost���;�4p_+       ��K	$a��A�*

logging/current_costЙ�;���s+       ��K	�Ca��A�*

logging/current_cost���;X�k'+       ��K	ua��A�*

logging/current_costG��;�$�+       ��K	��a��A�*

logging/current_costС�;���+       ��K	��a��A�*

logging/current_costu��;��s�+       ��K	��a��A�*

logging/current_costܗ�;���+       ��K	.b��A�*

logging/current_cost��;�|�&+       ��K	�]b��A�*

logging/current_cost2��;�"��+       ��K	͊b��A�*

logging/current_costP��;paܗ+       ��K	j�b��A�*

logging/current_cost���;�$�W+       ��K	��b��A�*

logging/current_cost���;(!�+       ��K	Lc��A�*

logging/current_cost���;Z�	�+       ��K	�Fc��A�*

logging/current_cost���;�,ҩ+       ��K	Yxc��A�*

logging/current_costp��;ys�+       ��K	W�c��A�*

logging/current_cost���;���+       ��K	Y�c��A�*

logging/current_cost���;�9~q+       ��K	A�c��A�*

logging/current_cost��;aM�+       ��K	',d��A�*

logging/current_cost���;h^��+       ��K	�Yd��A�*

logging/current_cost���;��jw+       ��K	E�d��A�*

logging/current_costg��;u4�-+       ��K	͵d��A�*

logging/current_costu�;$��.+       ��K	z�d��A�*

logging/current_cost,��;���{+       ��K	e��A�*

logging/current_cost	�;!n3+       ��K	+;e��A�*

logging/current_cost��;-��	+       ��K	�je��A�*

logging/current_cost��;��V�+       ��K	�e��A�*

logging/current_cost��;J�8Z+       ��K	\�e��A�*

logging/current_cost�1�;��.l+       ��K	%�e��A�*

logging/current_cost��;�o��+       ��K	�f��A�*

logging/current_cost�-�;A�b+       ��K	�Jf��A�*

logging/current_cost�7�;� �+       ��K	�wf��A�*

logging/current_costD�;=6�+       ��K	e�f��A�*

logging/current_costY2�;��&�+       ��K	�f��A�*

logging/current_costnV�;�IW�+       ��K	�g��A�*

logging/current_costyH�;�t�|+       ��K	)6g��A�*

logging/current_cost\>�;D�rU+       ��K	�cg��A�*

logging/current_costN\�;s&j+       ��K	h�g��A�*

logging/current_cost�Y�;��[y+       ��K	��g��A�*

logging/current_cost�A�;d<+�+       ��K	��g��A�*

logging/current_cost�c�;�5�+       ��K	h��A�*

logging/current_cost�p�;ګ��+       ��K	Hh��A�*

logging/current_cost�H�;����+       ��K	�vh��A�*

logging/current_cost�k�;�XRL+       ��K	�h��A�*

logging/current_costy~�;V��<+       ��K	��h��A�*

logging/current_cost<G�;)�+       ��K	�i��A�*

logging/current_cost�G�;��+       ��K	x0i��A�*

logging/current_costYv�;��+       ��K	�^i��A�*

logging/current_coste�;/r��+       ��K	V�i��A�*

logging/current_costDj�;�"�+       ��K	��i��A�*

logging/current_cost`��;y8��+       ��K	}�i��A�*

logging/current_cost$T�;Iߴm+       ��K	�j��A�*

logging/current_cost`h�;(��+       ��K	�Kj��A�*

logging/current_cost�4�;�A�+       ��K	Mxj��A�*

logging/current_costet�;y'�+       ��K	d�j��A�*

logging/current_cost�i�;��+       ��K	��j��A�*

logging/current_cost�3�;�X��+       ��K	 k��A�*

logging/current_cost|X�;����+       ��K	�2k��A�*

logging/current_cost�U�;k���+       ��K	`k��A�*

logging/current_cost�U�;m2E+       ��K	͏k��A�*

logging/current_cost2f�;�"#+       ��K	�k��A�*

logging/current_cost�I�;��(+       ��K	��k��A�*

logging/current_cost�d�;���+       ��K	�l��A�*

logging/current_cost�D�;9m��+       ��K	#Jl��A�*

logging/current_coste1�;�/��+       ��K	�{l��A�*

logging/current_cost�r�;�.�+       ��K	R�l��A�*

logging/current_cost7�;���+       ��K	��l��A�*

logging/current_cost�:�;,��+       ��K	Jm��A�*

logging/current_cost�7�;�y}+       ��K	�<m��A�*

logging/current_cost~%�;�o �+       ��K	km��A�*

logging/current_cost%,�;�4��+       ��K	��m��A�*

logging/current_cost�L�;��ߥ+       ��K	��m��A�*

logging/current_costk$�;��[+       ��K	:�m��A�*

logging/current_costF�; FGy+       ��K	�%n��A�*

logging/current_cost��;{�*�+       ��K	jVn��A�	*

logging/current_cost?�;�m�+       ��K	Շn��A�	*

logging/current_costn-�;�)�+       ��K	˴n��A�	*

logging/current_cost3�;�q�W+       ��K	��n��A�	*

logging/current_cost��;��/�+       ��K	�o��A�	*

logging/current_cost�:�;�]c+       ��K	Co��A�	*

logging/current_cost`!�;�1��+       ��K	�oo��A�	*

logging/current_cost�1�;hA�+       ��K	q�o��A�	*

logging/current_cost�&�;~+       ��K	�o��A�	*

logging/current_cost�-�;!�0g+       ��K	-�o��A�	*

logging/current_costR�;�i�g+       ��K	#+p��A�	*

logging/current_cost%O�;����+       ��K	�Xp��A�	*

logging/current_coste4�;���+       ��K	�p��A�	*

logging/current_costg��;�d
�+       ��K	�p��A�	*

logging/current_cost�;Y+       ��K		�p��A�	*

logging/current_cost�;�;�'�9+       ��K	9q��A�	*

logging/current_costIa�;�@��+       ��K	Dq��A�	*

logging/current_cost�k�;	�Ť+       ��K	!sq��A�	*

logging/current_cost�>�;�6$�+       ��K	ϡq��A�	*

logging/current_cost�m�;~L�+       ��K	��q��A�	*

logging/current_costR��;��+       ��K	k�q��A�	*

logging/current_cost�!�;/��+       ��K	�(r��A�	*

logging/current_cost�Y�;�vL'+       ��K	�Ur��A�	*

logging/current_costD��;�P��+       ��K	^�r��A�	*

logging/current_costb�;n�v�+       ��K	A�r��A�	*

logging/current_cost^l�;��&`+       ��K	��r��A�
*

logging/current_cost;]�;u�S.+       ��K	�
s��A�
*

logging/current_cost�0�;�`��+       ��K	8s��A�
*

logging/current_costtA�;Q�<�+       ��K	�es��A�
*

logging/current_cost�4�;��r�+       ��K	��s��A�
*

logging/current_cost0V�;ל�5+       ��K	��s��A�
*

logging/current_cost�4�;SyUS+       ��K	�s��A�
*

logging/current_cost�M�;��-+       ��K	�t��A�
*

logging/current_cost�R�;@�J�+       ��K	^Et��A�
*

logging/current_cost4�;��0+       ��K	�rt��A�
*

logging/current_cost^J�;ۣ�+       ��K	��t��A�
*

logging/current_cost@?�;՚;�+       ��K	�t��A�
*

logging/current_cost�+�;��3+       ��K	��t��A�
*

logging/current_costr"�;��w+       ��K	�(u��A�
*

logging/current_cost�*�;��a+       ��K	�Uu��A�
*

logging/current_cost��;���+       ��K	S�u��A�
*

logging/current_costF�;x�+       ��K	k�u��A�
*

logging/current_cost�T�;5n�+       ��K	0�u��A�
*

logging/current_costIB�;�K��+       ��K	7v��A�
*

logging/current_costy^�;�	U+       ��K	�5v��A�
*

logging/current_cost��;����+       ��K	{bv��A�
*

logging/current_cost܎�;����+       ��K	0�v��A�
*

logging/current_cost���;��a+       ��K	2�v��A�
*

logging/current_cost���;���+       ��K	D�v��A�
*

logging/current_cost���;B_+       ��K	# w��A�
*

logging/current_cost���;�FeX+       ��K	0Uw��A�
*

logging/current_cost���;P��+       ��K	��w��A�*

logging/current_cost\��;32�+       ��K	B�w��A�*

logging/current_cost9��;][S9+       ��K	x�w��A�*

logging/current_cost��;����+       ��K	�x��A�*

logging/current_costl�;����+       ��K	=x��A�*

logging/current_cost�,�;�C�[+       ��K	Mix��A�*

logging/current_cost�!�;0��~+       ��K	��x��A�*

logging/current_cost�7�;>1�+       ��K	��x��A�*

logging/current_cost�6�;M�8�+       ��K	��x��A�*

logging/current_cost3�;/�� +       ��K	<&y��A�*

logging/current_cost�5�;����+       ��K	�Ty��A�*

logging/current_cost�G�;�i �+       ��K	t�y��A�*

logging/current_costDH�;P4d+       ��K	��y��A�*

logging/current_cost�e�;�Z�e+       ��K	��y��A�*

logging/current_cost�O�;�
�+       ��K	z��A�*

logging/current_cost�]�;;w�j+       ��K	Ez��A�*

logging/current_cost�l�;qj)�+       ��K	Tpz��A�*

logging/current_cost+`�;6q�+       ��K	H�z��A�*

logging/current_costb�;����+       ��K	��z��A�*

logging/current_cost.}�;��<+       ��K	K {��A�*

logging/current_cost$��;` +       ��K	2.{��A�*

logging/current_cost���;�PI�+       ��K	�Z{��A�*

logging/current_cost�w�;d�H�+       ��K	1�{��A�*

logging/current_cost��;N�d+       ��K	��{��A�*

logging/current_costu��;���{+       ��K	S:|��A�*

logging/current_costٕ�;=ql+       ��K	�x|��A�*

logging/current_cost���;52��+       ��K	p�|��A�*

logging/current_cost¦�;+!�+       ��K	s�|��A�*

logging/current_cost��;z�u+       ��K	�$}��A�*

logging/current_cost���;����+       ��K	�\}��A�*

logging/current_cost��;I�v�+       ��K	W�}��A�*

logging/current_cost��;��9+       ��K	��}��A�*

logging/current_costy��;�M�+       ��K	��}��A�*

logging/current_cost���;3��Q+       ��K	�.~��A�*

logging/current_cost���;��{+       ��K	I_~��A�*

logging/current_cost���;�{1�+       ��K	m�~��A�*

logging/current_cost9��;C��+       ��K	R�~��A�*

logging/current_cost���;�E�R+       ��K	l�~��A�*

logging/current_costr��;u�޴+       ��K	0��A�*

logging/current_cost���;01�1+       ��K	�b��A�*

logging/current_cost���;Hn�|+       ��K	����A�*

logging/current_cost��;��~+       ��K	����A�*

logging/current_cost���;\���+       ��K	����A�*

logging/current_cost���;�b++       ��K	����A�*

logging/current_cost|��;�3m�+       ��K	`J���A�*

logging/current_costd��;�Ylm+       ��K	�|���A�*

logging/current_cost���;5PN+       ��K	&����A�*

logging/current_cost���;��P+       ��K	u݀��A�*

logging/current_cost\��;QI�+       ��K	,���A�*

logging/current_cost2��;+j�+       ��K	�A���A�*

logging/current_cost���;b�d+       ��K	�t���A�*

logging/current_costu��;���/+       ��K	L����A�*

logging/current_cost���;M��+       ��K	ҁ��A�*

logging/current_cost���;ސr\+       ��K	t����A�*

logging/current_cost���;��+       ��K	�,���A�*

logging/current_costG��;ٖ÷+       ��K	�f���A�*

logging/current_cost���;X�Y�+       ��K	�����A�*

logging/current_cost��;����+       ��K	����A�*

logging/current_cost���;d��+       ��K	����A�*

logging/current_costH�;첇�+       ��K	����A�*

logging/current_cost2��;Ŷg�+       ��K	�K���A�*

logging/current_costBx�;[�a[+       ��K	�z���A�*

logging/current_cost��;2d�+       ��K	�����A�*

logging/current_cost��;��x+       ��K	]Ӄ��A�*

logging/current_cost�;��4�+       ��K	�����A�*

logging/current_costN��;�g[�+       ��K	�.���A�*

logging/current_cost��;�=O+       ��K	wc���A�*

logging/current_cost���;�J��+       ��K	Ɠ���A�*

logging/current_cost\��;�)7�+       ��K	���A�*

logging/current_cost7��;��a+       ��K	����A�*

logging/current_cost���;�?+       ��K	����A�*

logging/current_costԳ�;���r+       ��K	SK���A�*

logging/current_costǿ�;�'o+       ��K	|���A�*

logging/current_cost���;�� +       ��K	:����A�*

logging/current_cost��;L +       ��K	�ޅ��A�*

logging/current_costD��;\r��+       ��K	����A�*

logging/current_cost���;]yaU+       ��K	�;���A�*

logging/current_cost���;�A�+       ��K	Nn���A�*

logging/current_cost��;�QN�+       ��K	X����A�*

logging/current_cost���;<?�+       ��K	%φ��A�*

logging/current_cost���;���+       ��K	�����A�*

logging/current_cost���;.f�+       ��K	�/���A�*

logging/current_cost���;�p+       ��K	�c���A�*

logging/current_cost���;>�+       ��K	�����A�*

logging/current_cost'��;��qa+       ��K	?����A�*

logging/current_costr��;ʚ�a+       ��K	R'���A�*

logging/current_cost5��;�2�W+       ��K	u���A�*

logging/current_costw��;"�Ⱥ+       ��K	&����A�*

logging/current_costK��;yX�`+       ��K	����A�*

logging/current_cost���;n��3+       ��K	q%���A�*

logging/current_cost0��;�.��+       ��K	Sl���A�*

logging/current_costǤ�;���+       ��K	ͫ���A�*

logging/current_cost���;��3�+       ��K	���A�*

logging/current_cost��;Pu&++       ��K	&���A�*

logging/current_cost��;��*e+       ��K	�X���A�*

logging/current_cost���;�^�+       ��K	"����A�*

logging/current_costN��;)��+       ��K	ʊ��A�*

logging/current_cost��;��++       ��K	Q����A�*

logging/current_cost���;�A0�+       ��K	�0���A�*

logging/current_cost���;�@+       ��K	�c���A�*

logging/current_cost��;rЋ�+       ��K	�����A�*

logging/current_cost���;�-�+       ��K	�����A�*

logging/current_cost��;/�7+       ��K	E���A�*

logging/current_cost�/�;�x�+       ��K	�!���A�*

logging/current_cost��;���+       ��K	KQ���A�*

logging/current_cost���;��e4+       ��K	h~���A�*

logging/current_costp�;h�i
+       ��K	����A�*

logging/current_cost��;��+       ��K	^،��A�*

logging/current_cost��;�?�D+       ��K	����A�*

logging/current_cost���;�fI�+       ��K	?:���A�*

logging/current_cost���;"�8*+       ��K	"k���A�*

logging/current_cost� �;B�V+       ��K	.����A�*

logging/current_costI �;���1+       ��K	Vʍ��A�*

logging/current_cost��;)	ў+       ��K	5����A�*

logging/current_cost��;�}M+       ��K	d4���A�*

logging/current_cost<�;�!e`+       ��K	Qq���A�*

logging/current_cost��;��Yn+       ��K	�����A�*

logging/current_costb�;T�+       ��K	^Ύ��A�*

logging/current_cost4�;n-�+       ��K	����A�*

logging/current_cost�B�;	-��+       ��K	@1���A�*

logging/current_cost-�;�>�9+       ��K	y`���A�*

logging/current_cost��;|�9�+       ��K	����A�*

logging/current_costK�;�N�?+       ��K	����A�*

logging/current_cost�$�;ˣY�+       ��K	`���A�*

logging/current_costI#�;�x#+       ��K	����A�*

logging/current_cost�*�;���:+       ��K	FI���A�*

logging/current_cost�`�;rbLw+       ��K	�v���A�*

logging/current_cost�c�;�x�+       ��K	;����A�*

logging/current_cost�:�;PN6+       ��K	_ѐ��A�*

logging/current_costr�;rq�-+       ��K	E����A�*

logging/current_cost9�;�Γ+       ��K	�3���A�*

logging/current_cost�`�;֓�"+       ��K	�g���A�*

logging/current_costtI�;��;b+       ��K	s����A�*

logging/current_cost57�;�:�s+       ��K	�đ��A�*

logging/current_cost�C�;��+       ��K	����A�*

logging/current_costIL�;��:+       ��K	%���A�*

logging/current_costgW�;��+       ��K	�P���A�*

logging/current_cost.T�;cS�+       ��K	�}���A�*

logging/current_costV�;�f�P+       ��K	����A�*

logging/current_costGc�;]��+       ��K	�ޒ��A�*

logging/current_cost�Y�;{�O+       ��K	9���A�*

logging/current_costnA�;f��+       ��K	�;���A�*

logging/current_cost�r�;ǀN+       ��K	Fm���A�*

logging/current_cost̥�;-�u+       ��K	�����A�*

logging/current_cost\��;Ru<+       ��K	�͓��A�*

logging/current_costG��;�]�+       ��K	�����A�*

logging/current_cost�y�;�^�+       ��K	�.���A�*

logging/current_cost"e�;��+       ��K	Se���A�*

logging/current_cost�d�;	륃+       ��K	�����A�*

logging/current_costLq�;���\+       ��K	Δ��A�*

logging/current_costk��;�Pz�+       ��K	�����A�*

logging/current_cost���;�*;�+       ��K	�*���A�*

logging/current_cost���;�G+       ��K	dV���A�*

logging/current_cost���;�հJ+       ��K	����A�*

logging/current_cost���;���[+       ��K	"����A�*

logging/current_cost%��;�E(#+       ��K	c���A�*

logging/current_cost���;**��+       ��K	d���A�*

logging/current_costl��;�El�+       ��K	�A���A�*

logging/current_costi��;�*+       ��K	zt���A�*

logging/current_cost`��;d��+       ��K	�����A�*

logging/current_costG��;��vY+       ��K	Ֆ��A�*

logging/current_cost$��;"ɑ+       ��K	����A�*

logging/current_cost��;��3�+       ��K	�.���A�*

logging/current_cost�u�;h�)+       ��K	�[���A�*

logging/current_cost���;��t1+       ��K	�����A�*

logging/current_cost�t�;�P��+       ��K	|����A�*

logging/current_cost�y�;�W�+       ��K	9���A�*

logging/current_cost|�;��+       ��K	����A�*

logging/current_cost�e�;\�g�+       ��K	^E���A�*

logging/current_cost��;����+       ��K	)s���A�*

logging/current_cost2��;��h�+       ��K	�����A�*

logging/current_cost��;���+       ��K	/Θ��A�*

logging/current_cost%��;�Av+       ��K	|����A�*

logging/current_cost5��;\��+       ��K	b(���A�*

logging/current_cost$�; �n�+       ��K	gV���A�*

logging/current_cost���;#�V+       ��K	����A�*

logging/current_cost��;_�`�+       ��K	6����A�*

logging/current_cost��;���z+       ��K	c���A�*

logging/current_cost%��;��}x+       ��K	q���A�*

logging/current_cost���;�`҂+       ��K	�B���A�*

logging/current_cost��;W3ݟ+       ��K	8p���A�*

logging/current_costΜ�;��V+       ��K	d����A�*

logging/current_costՖ�;����+       ��K	̚��A�*

logging/current_cost���;��k+       ��K	�����A�*

logging/current_cost`��;��C+       ��K	�$���A�*

logging/current_costR��;-�wy+       ��K	dQ���A�*

logging/current_cost���;J�z:+       ��K	�~���A�*

logging/current_costd��;�	Ǟ+       ��K	����A�*

logging/current_cost���;��(�+       ��K	�ٛ��A�*

logging/current_cost`��;Lh��+       ��K	���A�*

logging/current_cost���;���+       ��K	03���A�*

logging/current_cost���;�i�#+       ��K	�`���A�*

logging/current_cost��;�<ģ+       ��K	�����A�*

logging/current_cost���;b�c+       ��K	^����A�*

logging/current_cost��;13�%+       ��K	9���A�*

logging/current_costT��;�V��+       ��K	����A�*

logging/current_cost`��;܉��+       ��K	kK���A�*

logging/current_cost���;���9+       ��K	&x���A�*

logging/current_cost���;�G��+       ��K	U����A�*

logging/current_cost���;���+       ��K	�Н��A�*

logging/current_cost���;0�|D+       ��K	����A�*

logging/current_cost���;��|[+       ��K	l<���A�*

logging/current_costy��;��`
+       ��K	
k���A�*

logging/current_cost��;<'��+       ��K	�����A�*

logging/current_cost���;AϚ0+       ��K	}Ş��A�*

logging/current_costg��;�{2�+       ��K	V���A�*

logging/current_cost��;R��;+       ��K	_!���A�*

logging/current_cost$��;���+       ��K	�N���A�*

logging/current_cost���;���+       ��K	7{���A�*

logging/current_cost���;�p=+       ��K	~����A�*

logging/current_cost���;�^b+       ��K	u۟��A�*

logging/current_cost���;.��+       ��K	����A�*

logging/current_cost��;��S�+       ��K	]>���A�*

logging/current_cost���;�|�+       ��K	ls���A�*

logging/current_cost���;�/]1+       ��K	p����A�*

logging/current_cost��;W�-�+       ��K	7͠��A�*

logging/current_cost��;�`)�+       ��K	����A�*

logging/current_cost��;���+       ��K	m-���A�*

logging/current_cost���;�:=X+       ��K	�Z���A�*

logging/current_costR��;\K��+       ��K	�����A�*

logging/current_cost���;%�e+       ��K	%����A�*

logging/current_cost��;��+       ��K	����A�*

logging/current_cost���;��+       ��K	s���A�*

logging/current_cost���;�k�+       ��K	RE���A�*

logging/current_cost���;~�+       ��K	Mt���A�*

logging/current_coste��;C�7�+       ��K	)����A�*

logging/current_costd��;`�]b+       ��K	#Ӣ��A�*

logging/current_cost���;�W�+       ��K	6���A�*

logging/current_cost~��;m9�+       ��K	�1���A�*

logging/current_costp��;�Nq�+       ��K	�`���A�*

logging/current_cost|��;�O�+       ��K	s����A�*

logging/current_cost6�;��+       ��K	����A�*

logging/current_cost2\�;l��+       ��K	����A�*

logging/current_cost��;��,j+       ��K	����A�*

logging/current_cost��;?t��+       ��K	�F���A�*

logging/current_cost���; �!�+       ��K	�s���A�*

logging/current_cost��;@�Z�+       ��K	㢤��A�*

logging/current_cost ��;y���+       ��K	�Ϥ��A�*

logging/current_cost���;/���+       ��K	x����A�*

logging/current_cost���;*+       ��K	�,���A�*

logging/current_cost7��;��`�+       ��K	�\���A�*

logging/current_cost���;\���+       ��K	�����A�*

logging/current_cost��;�v	+       ��K	g����A�*

logging/current_cost�;tXu�+       ��K	9���A�*

logging/current_cost��;U��'+       ��K	����A�*

logging/current_costy��;Ev+       ��K	�>���A�*

logging/current_costu	�;����+       ��K	�n���A�*

logging/current_cost��;�Ml�+       ��K	Ĝ���A�*

logging/current_cost��;GR��+       ��K	�ʦ��A�*

logging/current_cost���;�O��+       ��K	`����A�*

logging/current_cost���;Z�<B+       ��K	-���A�*

logging/current_cost���;�E+       ��K	AZ���A�*

logging/current_cost���;]�0+       ��K	:����A�*

logging/current_cost���;r�+       ��K	¸���A�*

logging/current_cost<��;٬�+       ��K	���A�*

logging/current_costR�;�K�T+       ��K	����A�*

logging/current_cost;�;�
AQ+       ��K	�B���A�*

logging/current_cost��;mT-+       ��K	$n���A�*

logging/current_cost�.�;��x+       ��K	�����A�*

logging/current_cost�U�;�%�r+       ��K	�ʨ��A�*

logging/current_cost�7�;I
�P+       ��K	�����A�*

logging/current_cost�=�;h�4,+       ��K	�%���A�*

logging/current_cost`�;�+       ��K	�R���A�*

logging/current_cost���;�#�+       ��K	���A�*

logging/current_costb��;�n +       ��K	ү���A�*

logging/current_cost2
�;��70+       ��K	�ީ��A�*

logging/current_cost���;/߰+       ��K	����A�*

logging/current_costR�;��-4+       ��K	V8���A�*

logging/current_cost��;�'*a+       ��K	Re���A�*

logging/current_cost���;ۙ�Q+       ��K	M����A�*

logging/current_costK��;C(Ԛ+       ��K	N����A�*

logging/current_cost��;���+       ��K	���A�*

logging/current_costL�;xԩ+       ��K	����A�*

logging/current_cost�#�;�H�+       ��K	eT���A�*

logging/current_cost@!�;��A&+       ��K	r����A�*

logging/current_cost�!�;Q���+       ��K	۱���A�*

logging/current_cost\�;��+       ��K	����A�*

logging/current_cost)�;V���+       ��K	>���A�*

logging/current_costk�;tO�+       ��K	�@���A�*

logging/current_cost���;�O�+       ��K	�n���A�*

logging/current_cost+�;_b�+       ��K	e����A�*

logging/current_cost���;^̞�+       ��K	�Ȭ��A�*

logging/current_cost��;�l�+       ��K	b����A�*

logging/current_cost��;b��+       ��K	f'���A�*

logging/current_costg�;e��H+       ��K	|U���A�*

logging/current_cost� �;�F��+       ��K	@����A�*

logging/current_costE�;26I+       ��K	!����A�*

logging/current_cost��;�k�+       ��K	�߭��A�*

logging/current_cost��;��=+       ��K	F���A�*

logging/current_cost^
�;�4v+       ��K	�<���A�*

logging/current_cost;��;l��+       ��K	�l���A�*

logging/current_cost���;��w+       ��K	�����A�*

logging/current_cost���;��+       ��K	�Ȯ��A�*

logging/current_cost ��;�ms�+       ��K	�����A�*

logging/current_cost���;�܍F+       ��K	�&���A�*

logging/current_cost���;��,+       ��K	�T���A�*

logging/current_cost��;�a�2+       ��K	U����A�*

logging/current_cost��;�A�+       ��K	n����A�*

logging/current_cost��;��qi+       ��K	���A�*

logging/current_costE�;z�+       ��K	����A�*

logging/current_cost�;5� �+       ��K	<A���A�*

logging/current_cost��;�O��+       ��K	to���A�*

logging/current_cost�.�;8!��+       ��K	؜���A�*

logging/current_costD*�;�vy�+       ��K	�ʰ��A�*

logging/current_cost���; g��+       ��K	C����A�*

logging/current_cost��;��r+       ��K	2,���A�*

logging/current_cost���;H��+       ��K	pZ���A�*

logging/current_cost���;G�>�+       ��K	����A�*

logging/current_cost���;nd�+       ��K	P����A�*

logging/current_cost��;C�-�+       ��K	����A�*

logging/current_cost�!�;�|'�+       ��K	����A�*

logging/current_cost�%�;Et+       ��K	�;���A�*

logging/current_cost�E�;�@D�+       ��K	$i���A�*

logging/current_cost��;��̟+       ��K	p����A�*

logging/current_cost�A�;��t�+       ��K	+ò��A�*

logging/current_costr�;�h��+       ��K	����A�*

logging/current_costw�;�&�+       ��K	����A�*

logging/current_cost��;Xd?�+       ��K	�K���A�*

logging/current_cost��;N*+       ��K	z���A�*

logging/current_cost�6�;h�+       ��K	G����A�*

logging/current_costu�;�{=�+       ��K	@ѳ��A�*

logging/current_cost��;�]@+       ��K	�����A�*

logging/current_cost��;RX�+       ��K	�/���A�*

logging/current_costr��;����+       ��K	_���A�*

logging/current_cost��;\��+       ��K	�����A�*

logging/current_costi��;��C+       ��K	
����A�*

logging/current_cost�A�;����+       ��K	����A�*

logging/current_cost2>�;!�ܶ+       ��K	����A�*

logging/current_cost�P�;:맨+       ��K	�?���A�*

logging/current_costu�;�lz+       ��K	+n���A�*

logging/current_cost��;�Q�	+       ��K	l����A�*

logging/current_cost���;�Q�w+       ��K	�ȵ��A�*

logging/current_costg�;� �+       ��K	�����A�*

logging/current_cost��;	��s+       ��K	#���A�*

logging/current_cost�z�;����+       ��K	�Q���A�*

logging/current_cost<<�;U{�S+       ��K	����A�*

logging/current_cost���;��9+       ��K	~����A�*

logging/current_cost���;�$�+       ��K	�ڶ��A�*

logging/current_cost���;Z��+       ��K	����A�*

logging/current_cost��;�x�e+       ��K	~3���A�*

logging/current_cost���;���v+       ��K	�`���A�*

logging/current_cost4��;�a�+       ��K	#����A�*

logging/current_cost2��;r���+       ��K	�����A�*

logging/current_cost.��;��G+       ��K	b���A�*

logging/current_cost�y�;��-P+       ��K	���A�*

logging/current_cost,��;>l6�+       ��K	�G���A�*

logging/current_cost���;�|+       ��K	���A�*

logging/current_cost���;n��>+       ��K	2����A�*

logging/current_cost��;��"�+       ��K	�ݸ��A�*

logging/current_cost{��;���{+       ��K	S
���A�*

logging/current_cost���;Ȅ�p+       ��K	A;���A�*

logging/current_cost���;E�ks+       ��K	�i���A�*

logging/current_costT��;����+       ��K	.����A�*

logging/current_costN��;N��A+       ��K	�ɹ��A�*

logging/current_cost ��;*/w�+       ��K	�����A�*

logging/current_costd��;��f+       ��K	*���A�*

logging/current_costε�;�]@�+       ��K	�W���A�*

logging/current_costN��;:f�7+       ��K	���A�*

logging/current_cost���;����+       ��K	;����A�*

logging/current_cost���;����+       ��K	����A�*

logging/current_cost`��;e��>+       ��K	D���A�*

logging/current_cost���;�cЎ+       ��K	 B���A�*

logging/current_cost��;���+       ��K	�����A�*

logging/current_cost���;Q���+       ��K	�ֻ��A�*

logging/current_cost4��;;^З+       ��K	����A�*

logging/current_costܲ�;�1�"+       ��K	e7���A�*

logging/current_cost���;bP�O+       ��K	Bc���A�*

logging/current_cost��;�(t�+       ��K	e����A�*

logging/current_cost���;w7z+       ��K	�����A�*

logging/current_costk��;`���+       ��K	���A�*

logging/current_costE��;S�XO+       ��K	j"���A�*

logging/current_cost��;�#�+       ��K	�Q���A�*

logging/current_cost���;KO4T+       ��K	S����A�*

logging/current_cost��;	��i+       ��K	M����A�*

logging/current_cost�x�;��}�+       ��K	�ܽ��A�*

logging/current_cost���;�Q1,+       ��K	����A�*

logging/current_costb�;�A�+       ��K	�<���A�*

logging/current_cost���;D�3Q+       ��K	�m���A�*

logging/current_costl�;�Ƒ +       ��K	ʜ���A�*

logging/current_cost@�;V��9+       ��K	�ɾ��A�*

logging/current_cost��;���+       ��K	�����A�*

logging/current_cost�	�;~�^]+       ��K	�'���A�*

logging/current_cost`5�;efd�+       ��K	�Y���A�*

logging/current_costTb�;���+       ��K	ㇿ��A�*

logging/current_costpn�;�J��+       ��K	����A�*

logging/current_cost�-�;?�<+       ��K	�߿��A�*

logging/current_costY�;���+       ��K	����A�*

logging/current_costl��;l}`5+       ��K	�;���A�*

logging/current_cost��;8�nX+       ��K	�i���A�*

logging/current_costG��;Đ��+       ��K	C����A�*

logging/current_costl
�;���K+       ��K	L����A�*

logging/current_cost��;�Zr+       ��K	3����A�*

logging/current_cost���;w��+       ��K	� ���A�*

logging/current_cost���;���+       ��K	_f���A�*

logging/current_costg��;�Xi+       ��K	����A�*

logging/current_cost���;�+yH+       ��K	N����A�*

logging/current_cost���;�J+       ��K	k���A�*

logging/current_cost���;�2��+       ��K	}Q���A�*

logging/current_cost	��;��|+       ��K	݇���A�*

logging/current_cost`��;bMwN+       ��K	�����A�*

logging/current_cost���;���+       ��K	;
���A�*

logging/current_cost	��;�W2+       ��K	�F���A�*

logging/current_costR��;��*-+       ��K	W����A�*

logging/current_costE��;.n>t+       ��K	&����A�*

logging/current_cost��;h���+       ��K	%����A�*

logging/current_costĮ�;:+       ��K	�.���A�*

logging/current_costUx�;�+       ��K	bi���A�*

logging/current_cost���;�)ĳ+       ��K	5����A�*

logging/current_cost��;�+       ��K	�����A�*

logging/current_costٚ�;k�!�+       ��K	����A�*

logging/current_cost���;T�è+       ��K	�2���A�*

logging/current_cost���;����+       ��K	�b���A�*

logging/current_cost���;.!?+       ��K	s����A�*

logging/current_cost���;��i�+       ��K	�����A�*

logging/current_cost��;,˩
+       ��K	W���A�*

logging/current_cost`��;�<�+       ��K	p4���A�*

logging/current_cost)��;,M�+       ��K	�g���A�*

logging/current_costY��;Ry�+       ��K	����A�*

logging/current_costо�;���+       ��K	�����A�*

logging/current_cost���;�rռ+       ��K	�����A�*

logging/current_cost�l�;f�S�+       ��K	�-���A�*

logging/current_cost���;Q��G+       ��K	�[���A�*

logging/current_cost���;����+       ��K	����A�*

logging/current_costb��;���+       ��K	�����A�*

logging/current_cost�q�;f�)+       ��K	�����A�*

logging/current_cost���;�|+       ��K	�0���A�*

logging/current_cost���;W��+       ��K	]b���A�*

logging/current_costײ�;�n;F+       ��K	����A�*

logging/current_cost���;ͩ�+       ��K	����A�*

logging/current_cost\��;˷2+       ��K	�����A�*

logging/current_cost4��;����+       ��K	!���A�*

logging/current_cost���;n��+       ��K	uQ���A�*

logging/current_cost���;�{`+       ��K	�}���A�*

logging/current_cost4��;�l4M+       ��K	����A�*

logging/current_cost���;�b*�+       ��K	�����A�*

logging/current_costɜ�;��� +       ��K	����A�*

logging/current_cost���;T��+       ��K	<K���A�*

logging/current_cost�x�;�<�+       ��K	dw���A�*

logging/current_cost���;Ç~y+       ��K	�����A�*

logging/current_cost���;0���+       ��K	�����A�*

logging/current_costK}�;(.J�+       ��K	����A�*

logging/current_costę�;���+       ��K	71���A�*

logging/current_cost��;��x+       ��K	_���A�*

logging/current_costǕ�;���+       ��K	�����A�*

logging/current_cost��;yQ͛+       ��K	����A�*

logging/current_cost�f�;.���+       ��K	�����A�*

logging/current_cost�R�;���+       ��K	���A�*

logging/current_cost�T�;7��x+       ��K	GH���A�*

logging/current_costB��;�=ʪ+       ��K	�u���A�*

logging/current_cost�;^���+       ��K	����A�*

logging/current_costN�;ܿ�I+       ��K	����A�*

logging/current_cost�e�;�&&B+       ��K	�����A�*

logging/current_cost���;o��+       ��K	&-���A�*

logging/current_cost��;�D�q+       ��K	�Z���A�*

logging/current_cost@�;�o�:+       ��K	 ����A�*

logging/current_cost<)�;�M�+       ��K	�����A�*

logging/current_costn�;�j�+       ��K	c����A�*

logging/current_cost.�;D]H�+       ��K	T���A�*

logging/current_cost|"�;��1e+       ��K	jL���A�*

logging/current_cost"�;7�&+       ��K	U{���A�*

logging/current_cost%��;��HF+       ��K	%����A�*

logging/current_cost�9�;g}�+       ��K	�����A�*

logging/current_cost�7�;��p�+       ��K	k���A�*

logging/current_costb��;�-�+       ��K	�<���A�*

logging/current_costkW�;���+       ��K	�j���A�*

logging/current_cost���;��K�+       ��K	����A�*

logging/current_cost��;��3+       ��K	�����A�*

logging/current_cost���;����+       ��K	�����A�*

logging/current_cost,p�;��+       ��K	1#���A�*

logging/current_costi�;X�6l+       ��K	�S���A�*

logging/current_cost7��;�d�`+       ��K	B����A�*

logging/current_costUu�;�?�+       ��K	����A�*

logging/current_cost���;d���+       ��K	�����A�*

logging/current_cost�O�;%&�+       ��K	r���A�*

logging/current_cost�h�;#�nz+       ��K	>���A�*

logging/current_cost��;)���+       ��K	�l���A�*

logging/current_cost�V�;�4zy+       ��K	�����A�*

logging/current_cost |�;�[N�+       ��K	�����A�*

logging/current_cost�L�;l=�+       ��K	�����A�*

logging/current_cost�'�;*���+       ��K	�$���A�*

logging/current_cost�c�; �(�+       ��K	rS���A�*

logging/current_cost�(�;�ܧ+       ��K	3����A�*

logging/current_cost|V�;�N�{+       ��K	N����A�*

logging/current_cost�z�;(���+       ��K	�����A�*

logging/current_cost�*�;2��+       ��K	����A�*

logging/current_costY�;��N�+       ��K	�A���A�*

logging/current_cost�'�;�Լ�+       ��K	�u���A�*

logging/current_costE��;��G+       ��K	ǣ���A�*

logging/current_cost��;5���+       ��K	r����A�*

logging/current_costt�;�u{`+       ��K	�����A�*

logging/current_costt_�;W�S�+       ��K	�/���A�*

logging/current_cost<�;��V+       ��K	_���A�*

logging/current_costS�;ޑ�<+       ��K	�����A�*

logging/current_cost�+�;��&+       ��K	#����A�*

logging/current_cost�S�;>�+       ��K	k����A�*

logging/current_cost�m�;�E&N+       ��K	����A�*

logging/current_cost`��;�k�+       ��K	�E���A�*

logging/current_cost�7�;�qBv+       ��K	Pw���A�*

logging/current_cost��;Y�%h+       ��K	j����A�*

logging/current_cost9J�;�Q��+       ��K	�����A�*

logging/current_cost~i�;ي��+       ��K	M
���A�*

logging/current_cost$��;,�aC+       ��K	6���A�*

logging/current_cost��;�d�+       ��K	�h���A�*

logging/current_cost�<�;�Ff	+       ��K	����A�*

logging/current_cost���;���+       ��K	\����A�*

logging/current_cost��;(d?+       ��K	B����A�*

logging/current_cost\ �;�۟O+       ��K	�#���A�*

logging/current_costr,�;ߔp�+       ��K	Q���A�*

logging/current_cost��;���+       ��K	����A�*

logging/current_costT��;���+       ��K	m����A�*

logging/current_costkH�;IC +       ��K	�����A�*

logging/current_cost���;x��E+       ��K	I
���A�*

logging/current_cost���;v�D�+       ��K	W7���A�*

logging/current_cost2C�;��Q5+       ��K	�f���A�*

logging/current_costr��;n���+       ��K	.����A�*

logging/current_cost���;6��+       ��K	����A�*

logging/current_costiR�;��+       ��K	b����A�*

logging/current_cost��;�iю+       ��K	@$���A�*

logging/current_costP&�;��"+       ��K	�O���A�*

logging/current_cost	��;�
&+       ��K	�z���A�*

logging/current_costl�;�*u+       ��K	~����A�*

logging/current_cost7�;��_+       ��K	�����A�*

logging/current_cost'@�;sO�+       ��K	Q���A�*

logging/current_cost� �;sth�+       ��K	/���A�*

logging/current_cost �;Ҿ/�+       ��K	�]���A�*

logging/current_cost�a�;[}�+       ��K	;����A�*

logging/current_costy�;�LG�+       ��K	i����A�*

logging/current_cost�;�;r]�k+       ��K	`����A�*

logging/current_cost'*�;g�y+       ��K	����A�*

logging/current_cost��;��+       ��K	^@���A�*

logging/current_cost<,�;���+       ��K	Dm���A�*

logging/current_costDI�;��+       ��K	ɛ���A�*

logging/current_cost�	�;Qӥn+       ��K	�����A�*

logging/current_costKk�;�g+       ��K	����A�*

logging/current_cost�U�;��b+       ��K	Q$���A�*

logging/current_cost���;4z�+       ��K	�P���A�*

logging/current_cost�.�;����+       ��K	;���A�*

logging/current_cost�g�;�_�+       ��K	٬���A�*

logging/current_cost� �;��hc+       ��K	�����A� *

logging/current_cost��;x�v+       ��K	����A� *

logging/current_cost�=�;5��+       ��K	=���A� *

logging/current_cost��;�x+       ��K	li���A� *

logging/current_cost�V�;Clp+       ��K	���A� *

logging/current_cost�!�;f8�++       ��K	�����A� *

logging/current_costr%�;ͧ+       ��K	t����A� *

logging/current_cost�_�;/+       ��K	r#���A� *

logging/current_cost|9�;�Z��+       ��K	�P���A� *

logging/current_cost?�;p9��+       ��K	����A� *

logging/current_cost�(�;�j�+       ��K	֬���A� *

logging/current_cost���;�Sr+       ��K	�����A� *

logging/current_costn�;�L��+       ��K	�	���A� *

logging/current_costR�;,#�3+       ��K	�8���A� *

logging/current_cost���;�j�o+       ��K	�f���A� *

logging/current_cost�
�;�B+       ��K	����A� *

logging/current_costW��;��+       ��K	s����A� *

logging/current_costri�;2�h�+       ��K	O����A� *

logging/current_cost��;�Ʃ�+       ��K	+���A� *

logging/current_cost�5�;�p�+       ��K	tY���A� *

logging/current_cost�@�;@�+�+       ��K	Z����A� *

logging/current_cost�w�;�iM�+       ��K	B����A� *

logging/current_cost���;t�t+       ��K	����A� *

logging/current_cost���;
!M�+       ��K	"���A� *

logging/current_coste4�;�n��+       ��K	t?���A� *

logging/current_cost���;��#+       ��K	�s���A�!*

logging/current_costR��;97(�+       ��K	����A�!*

logging/current_cost5��;��x+       ��K	�����A�!*

logging/current_cost��;2�Yq+       ��K	s���A�!*

logging/current_cost��;Z,z�+       ��K	�3���A�!*

logging/current_cost��;���+       ��K	la���A�!*

logging/current_cost���;e��+       ��K	n����A�!*

logging/current_cost���;e̴Q+       ��K	�����A�!*

logging/current_cost'��;�+       ��K	�����A�!*

logging/current_cost ��;_��+       ��K	���A�!*

logging/current_cost&�;%�Κ+       ��K	eJ���A�!*

logging/current_cost��;5��+       ��K	W|���A�!*

logging/current_cost7�;;%�0+       ��K	�����A�!*

logging/current_cost��;
��+       ��K	
����A�!*

logging/current_cost�;�;��+       ��K	���A�!*

logging/current_cost�K�;o���+       ��K	�8���A�!*

logging/current_costku�;3�T+       ��K	ti���A�!*

logging/current_costyc�;A��+       ��K	N����A�!*

logging/current_cost��;X��+       ��K	�����A�!*

logging/current_costҟ�;�
��+       ��K	����A�!*

logging/current_cost5R�;a�W�+       ��K	!���A�!*

logging/current_costU�;�y��+       ��K	�M���A�!*

logging/current_cost[q�;[F��+       ��K	�}���A�!*

logging/current_costG!�;����+       ��K	/����A�!*

logging/current_cost|��;�j�@+       ��K	����A�!*

logging/current_cost���;[+�+       ��K	����A�!*

logging/current_cost�G�;j:)�+       ��K	�:���A�"*

logging/current_cost���;O>|:+       ��K	.h���A�"*

logging/current_cost�z�;VQ�l+       ��K	×���A�"*

logging/current_cost7�;���+       ��K	����A�"*

logging/current_costՏ�;��+       ��K	;����A�"*

logging/current_cost4_�;@&��+       ��K	�$���A�"*

logging/current_costg��;��+       ��K	�P���A�"*

logging/current_cost��;��X+       ��K	$����A�"*

logging/current_cost��;�Fs'+       ��K	v����A�"*

logging/current_cost�!�;
s%l+       ��K	�����A�"*

logging/current_costK��;rY��+       ��K	M���A�"*

logging/current_cost���;�C��+       ��K	K9���A�"*

logging/current_cost��;d
G6+       ��K	�f���A�"*

logging/current_cost`b�;��/�+       ��K	!����A�"*

logging/current_cost���;�~w&+       ��K	�����A�"*

logging/current_cost�e�;�#vQ+       ��K	I����A�"*

logging/current_costdx�;�.đ+       ��K	�!���A�"*

logging/current_cost{��;�M�B+       ��K	�N���A�"*

logging/current_cost2z�;��+       ��K	4{���A�"*

logging/current_cost"�;l)�+       ��K	o����A�"*

logging/current_cost���;�+       ��K	����A�"*

logging/current_cost ��;
���+       ��K	����A�"*

logging/current_cost��;��+       ��K	�5���A�"*

logging/current_cost��;�f�+       ��K	e���A�"*

logging/current_costg�;�<�+       ��K	1����A�"*

logging/current_costb�;�L��+       ��K	����A�#*

logging/current_costR3�;OE�-+       ��K	[����A�#*

logging/current_costn�;5�+       ��K	?���A�#*

logging/current_costD$�;���w+       ��K	QK���A�#*

logging/current_cost��;���t+       ��K	Tw���A�#*

logging/current_cost\q�;���s+       ��K	ͤ���A�#*

logging/current_cost�E�;F-�+       ��K	;����A�#*

logging/current_costul�;\�9�+       ��K	�����A�#*

logging/current_cost���;�u1�+       ��K	�+���A�#*

logging/current_costD��;�b�4+       ��K	�W���A�#*

logging/current_cost��;�`H-+       ��K	̈́���A�#*

logging/current_cost��;	D�S+       ��K	R����A�#*

logging/current_costK��;b�c�+       ��K	�����A�#*

logging/current_cost���;����+       ��K	����A�#*

logging/current_cost���;v�+       ��K	_C���A�#*

logging/current_cost��;I��+       ��K	�q���A�#*

logging/current_costRN�;��+       ��K	����A�#*

logging/current_costG(�;�}@=+       ��K	R����A�#*

logging/current_cost<m�;9M��+       ��K	�����A�#*

logging/current_cost��;�F��+       ��K	i)���A�#*

logging/current_cost���;���I+       ��K	�U���A�#*

logging/current_cost�"�;`���+       ��K	ք���A�#*

logging/current_cost�v�;L�k'+       ��K	����A�#*

logging/current_cost�2�;�/$�+       ��K	X����A�#*

logging/current_costN#�;���+       ��K	X���A�#*

logging/current_cost �;�W�,+       ��K	I=���A�#*

logging/current_cost���;{K�{+       ��K	�k���A�$*

logging/current_cost2��;�UI+       ��K	k����A�$*

logging/current_cost���;,�0�+       ��K	>����A�$*

logging/current_cost`��;q�k+       ��K	�����A�$*

logging/current_cost���;$X~�+       ��K	&(���A�$*

logging/current_cost���;��w$+       ��K	�W���A�$*

logging/current_cost;l�;�K��+       ��K	O����A�$*

logging/current_costU��;i5�x+       ��K	(����A�$*

logging/current_cost)�;5��+       ��K	�����A�$*

logging/current_cost|�;�N'!+       ��K	����A�$*

logging/current_cost���;�q(�+       ��K	>F���A�$*

logging/current_cost�a�;��x+       ��K	Nv���A�$*

logging/current_cost���;�,+       ��K	����A�$*

logging/current_costUI�;pǕn+       ��K	�����A�$*

logging/current_cost\��;����+       ��K	�����A�$*

logging/current_cost��;h�ڈ+       ��K	�.���A�$*

logging/current_cost���;�F�	+       ��K	�]���A�$*

logging/current_cost2}�;��ެ+       ��K	e����A�$*

logging/current_cost+1�;��+       ��K	8����A�$*

logging/current_cost��;��<V+       ��K	�����A�$*

logging/current_cost���;��X�+       ��K	����A�$*

logging/current_costlJ�;[i�+       ��K	UB���A�$*

logging/current_cost���;G���+       ��K	�p���A�$*

logging/current_cost.�;#E��+       ��K	%����A�$*

logging/current_cost���;�*�+       ��K	0����A�$*

logging/current_costl��;2j~�+       ��K	�����A�$*

logging/current_costٵ�;#mc+       ��K	(���A�%*

logging/current_cost�'�;9_?�+       ��K	LV���A�%*

logging/current_cost ��;�o�+       ��K	b����A�%*

logging/current_cost���;�n�>+       ��K	�����A�%*

logging/current_costl9�;.ѐ�+       ��K	�����A�%*

logging/current_cost�A�;/�+       ��K	���A�%*

logging/current_cost 4�;omN6+       ��K	p:���A�%*

logging/current_cost�Q�;7�ݖ+       ��K	}h���A�%*

logging/current_cost"(�;�F �+       ��K	1����A�%*

logging/current_cost�>�;��g�+       ��K	�����A�%*

logging/current_costU��;�CKP+       ��K	s����A�%*

logging/current_costrM�;+j+       ��K	����A�%*

logging/current_costL��;�<EN+       ��K	�J���A�%*

logging/current_cost���;<�g�+       ��K	wx���A�%*

logging/current_cost�#�;�g+       ��K	\����A�%*

logging/current_cost.s�;�~�E+       ��K	]����A�%*

logging/current_cost��;#��+       ��K	+ ���A�%*

logging/current_costY��;��-�+       ��K	q.���A�%*

logging/current_cost��;kD+       ��K	\���A�%*

logging/current_cost���;��%+       ��K	�����A�%*

logging/current_cost���;JY�O+       ��K	�����A�%*

logging/current_cost��;T&�,+       ��K	�����A�%*

logging/current_cost���;*d&�+       ��K	u���A�%*

logging/current_cost2�;p�N�+       ��K	�G���A�%*

logging/current_cost���;�xo+       ��K	Rt���A�%*

logging/current_costT?�;>�|+       ��K	�����A�&*

logging/current_costd��;z��+       ��K	�����A�&*

logging/current_costY��;��;+       ��K	�����A�&*

logging/current_cost��;����+       ��K	b)���A�&*

logging/current_cost��;!&�U+       ��K	�U���A�&*

logging/current_cost@��;�K�[+       ��K	�����A�&*

logging/current_cost��;]�+       ��K	)����A�&*

logging/current_costE��;�1+       ��K	�����A�&*

logging/current_cost���;�6P�+       ��K	���A�&*

logging/current_costuN�;z�k+       ��K	L?���A�&*

logging/current_cost�4�;�[�+       ��K	�m���A�&*

logging/current_cost�-�;f��+       ��K	����A�&*

logging/current_cost�;m4�+       ��K	����A�&*

logging/current_cost���;�&̎+       ��K	�����A�&*

logging/current_cost�y�;y��+       ��K	H%���A�&*

logging/current_cost�K�;X+I+       ��K	CS���A�&*

logging/current_cost���;=�͌+       ��K	ݫ���A�&*

logging/current_costL��;�>�e+       ��K	�����A�&*

logging/current_costU��;&7��+       ��K	�#���A�&*

logging/current_cost���;�?��+       ��K	}_���A�&*

logging/current_cost��;%o+       ��K	����A�&*

logging/current_cost���;ې>B+       ��K	g����A�&*

logging/current_cost ��;���+       ��K	#���A�&*

logging/current_cost*�;F`H�+       ��K	�[���A�&*

logging/current_cost�R�;��m�+       ��K	�����A�&*

logging/current_cost.=�;A�y+       ��K	W����A�&*

logging/current_cost��;V��=+       ��K	���A�'*

logging/current_cost���;�~��+       ��K	�C���A�'*

logging/current_cost���;�d��+       ��K	����A�'*

logging/current_cost���;jǐQ+       ��K	�����A�'*

logging/current_cost�R�;i��+       ��K	�����A�'*

logging/current_cost|(�;�B�7+       ��K	e0���A�'*

logging/current_costޠ�;Ĭ�+       ��K	�h���A�'*

logging/current_costΚ�;�g��+       ��K	`����A�'*

logging/current_cost� �;	8�W+       ��K	�����A�'*

logging/current_cost �;$_��+       ��K	{
 ��A�'*

logging/current_cost�k�;�M�2+       ��K	(; ��A�'*

logging/current_cost���;u�&�+       ��K	�p ��A�'*

logging/current_costd<�;�^�+       ��K	� ��A�'*

logging/current_cost��;X~��+       ��K	$� ��A�'*

logging/current_cost���;p�ld+       ��K	}��A�'*

logging/current_costGp�;3�|+       ��K	�D��A�'*

logging/current_cost+��;w_�+       ��K	'y��A�'*

logging/current_cost2��;[5?�+       ��K	����A�'*

logging/current_cost>�;]'�t+       ��K	����A�'*

logging/current_cost���;�fg+       ��K	B��A�'*

logging/current_cost��;W)G�+       ��K	AH��A�'*

logging/current_cost��;b��x+       ��K	y~��A�'*

logging/current_costG@�;/}+       ��K	(���A�'*

logging/current_cost��;2��+       ��K	Y���A�'*

logging/current_cost5��;�tC+       ��K	���A�'*

logging/current_cost�~�;��L]+       ��K	%;��A�(*

logging/current_cost'��;\��	+       ��K	�j��A�(*

logging/current_cost���;+H�]+       ��K	����A�(*

logging/current_cost� �;5�V+       ��K	����A�(*

logging/current_cost<m�;�$X�+       ��K	����A�(*

logging/current_cost+%�;T�+       ��K	^��A�(*

logging/current_cost���;nS��+       ��K	aM��A�(*

logging/current_cost���;ᖐ++       ��K	���A�(*

logging/current_costu�;ծlk+       ��K	9���A�(*

logging/current_cost	{�;�'�`+       ��K	����A�(*

logging/current_cost��;��f!+       ��K	&	��A�(*

logging/current_cost���;gmZ+       ��K	�6��A�(*

logging/current_costҮ�;o|�+       ��K		f��A�(*

logging/current_cost�V�;��U�+       ��K	����A�(*

logging/current_cost._�;����+       ��K	)���A�(*

logging/current_cost��;���+       ��K	n���A�(*

logging/current_cost�]�;�Q�#+       ��K	���A�(*

logging/current_cost��;�F�w+       ��K	�I��A�(*

logging/current_cost��;KO+       ��K	�w��A�(*

logging/current_cost���;w|Yr+       ��K	L���A�(*

logging/current_costl��;���S+       ��K	����A�(*

logging/current_costY��;��Oz+       ��K	���A�(*

logging/current_cost��;]L>+       ��K	�1��A�(*

logging/current_cost�f�;8� i+       ��K	c��A�(*

logging/current_costKf�;��<+       ��K	���A�(*

logging/current_cost�U�;���+       ��K	u���A�(*

logging/current_cost��;���+       ��K	E���A�)*

logging/current_costb��;�'�*+       ��K	0 ��A�)*

logging/current_cost��;���+       ��K	sM��A�)*

logging/current_cost���;,d��+       ��K	���A�)*

logging/current_cost��;�4�_+       ��K	����A�)*

logging/current_cost"s�;W�|+       ��K	����A�)*

logging/current_costNN�;�e��+       ��K	>	��A�)*

logging/current_cost_�;��v�+       ��K	SI	��A�)*

logging/current_cost`�;w��+       ��K	�w	��A�)*

logging/current_cost�Y�;+%t�+       ��K	m�	��A�)*

logging/current_costǑ�;:���+       ��K	�	��A�)*

logging/current_cost ��;WE��+       ��K	
��A�)*

logging/current_costPU�;ّ�u+       ��K	I
��A�)*

logging/current_cost�;�;���+       ��K	Iy
��A�)*

logging/current_cost� �;N3+       ��K	u�
��A�)*

logging/current_cost���;ނ5Q+       ��K	}�
��A�)*

logging/current_cost�a�;p~hl+       ��K	���A�)*

logging/current_cost�u�;(Z� +       ��K	�;��A�)*

logging/current_costy?�;���y+       ��K	wj��A�)*

logging/current_cost"��;���e+       ��K	����A�)*

logging/current_cost���;�3�+       ��K	k���A�)*

logging/current_cost��;f��+       ��K	����A�)*

logging/current_cost.*�;���+       ��K	�)��A�)*

logging/current_costn�;����+       ��K	2V��A�)*

logging/current_costC�;�>&^+       ��K	Ȅ��A�)*

logging/current_cost���;-}*�+       ��K	C���A�)*

logging/current_cost��;
Z��+       ��K	S���A�**

logging/current_cost���;�6	�+       ��K	B��A�**

logging/current_costE�;��*+       ��K	�I��A�**

logging/current_cost`��;3�+       ��K	Zx��A�**

logging/current_cost�g�;}ڴ+       ��K	@���A�**

logging/current_costy�;I:S+       ��K	����A�**

logging/current_costR��;���+       ��K	���A�**

logging/current_cost���;�_'+       ��K	3E��A�**

logging/current_cost	j�;��<+       ��K	Ն��A�**

logging/current_cost�;ރ�B+       ��K	����A�**

logging/current_costk��;���+       ��K	���A�**

logging/current_cost�5�;+       ��K	O��A�**

logging/current_cost���;��q*+       ��K	l���A�**

logging/current_cost$T�;)]�+       ��K	J���A�**

logging/current_cost$c�;�8�+       ��K	���A�**

logging/current_costWP�;�t+       ��K	�A��A�**

logging/current_cost5^�;6ij�+       ��K	q��A�**

logging/current_cost���;*�.�+       ��K	^���A�**

logging/current_cost�5�;m�VS+       ��K	���A�**

logging/current_cost��;�v��+       ��K	���A�**

logging/current_costUo�;�lWY+       ��K	�M��A�**

logging/current_cost�D�;{7+       ��K	�~��A�**

logging/current_costn��;�P+       ��K	Ӯ��A�**

logging/current_costB��;|-^O+       ��K	����A�**

logging/current_costG�;��QE+       ��K	�#��A�**

logging/current_cost��;�l��+       ��K	�Z��A�+*

logging/current_cost'!�;��r�+       ��K	h���A�+*

logging/current_cost�;��7�+       ��K	Y���A�+*

logging/current_costY��;��"+       ��K	\���A�+*

logging/current_cost��;b���+       ��K	S��A�+*

logging/current_cost{��;{�r�+       ��K	�I��A�+*

logging/current_cost��;;P+       ��K	�{��A�+*

logging/current_cost��;̬��+       ��K	^���A�+*

logging/current_cost��;���	+       ��K	����A�+*

logging/current_cost��;=�I+       ��K	���A�+*

logging/current_cost�?�;^s�+       ��K	�k��A�+*

logging/current_cost�G�;����+       ��K	"���A�+*

logging/current_cost>��;˃+       ��K	����A�+*

logging/current_cost��;�IS+       ��K	���A�+*

logging/current_cost^��;^nFx+       ��K	�<��A�+*

logging/current_cost���;�=�
+       ��K	�k��A�+*

logging/current_costD��;q��<+       ��K	����A�+*

logging/current_cost���;V4ܛ+       ��K	����A�+*

logging/current_costGY�;���+       ��K	���A�+*

logging/current_cost�9�;��k"+       ��K	�1��A�+*

logging/current_cost���;��<B+       ��K	a^��A�+*

logging/current_cost˽�;g�п+       ��K		���A�+*

logging/current_cost@Q�;Y���+       ��K	4���A�+*

logging/current_costε�;��;+       ��K	����A�+*

logging/current_costĿ�;�~Z�+       ��K	���A�+*

logging/current_cost�|�;t��+       ��K	�N��A�+*

logging/current_cost˛�;#�8+       ��K	�{��A�,*

logging/current_costY��;^L�+       ��K	���A�,*

logging/current_cost�6�;��;�+       ��K	S���A�,*

logging/current_cost�%�;�Y��+       ��K	D��A�,*

logging/current_cost�n�;}D�+       ��K	�;��A�,*

logging/current_cost��;��n�+       ��K	�g��A�,*

logging/current_cost'{�;��UU+       ��K	;���A�,*

logging/current_cost��;���+       ��K	����A�,*

logging/current_cost�5�;� ]+       ��K	T���A�,*

logging/current_cost�^�;:�
+       ��K	e!��A�,*

logging/current_cost�i�;��-�+       ��K	rO��A�,*

logging/current_cost���;L��+       ��K	�|��A�,*

logging/current_costL�;��o!+       ��K	����A�,*

logging/current_costЬ�;�^��+       ��K	����A�,*

logging/current_cost���;����+       ��K	�
��A�,*

logging/current_cost ��;"k�+       ��K	�9��A�,*

logging/current_cost�~�;�k��+       ��K	�f��A�,*

logging/current_costG��;R_Pe+       ��K	6���A�,*

logging/current_cost���;Bm+       ��K	V���A�,*

logging/current_cost7�;N+       ��K	u���A�,*

logging/current_costL�;�`�h+       ��K	�.��A�,*

logging/current_cost\��;:���+       ��K	�\��A�,*

logging/current_costP�;J�I+       ��K	���A�,*

logging/current_cost���; ��:+       ��K	|���A�,*

logging/current_costr#�;.٪+       ��K	����A�,*

logging/current_cost���;�{N+       ��K	��A�-*

logging/current_cost���;�S�+       ��K	DE��A�-*

logging/current_cost���;Ծn+       ��K	ep��A�-*

logging/current_cost�O�;N�+       ��K	���A�-*

logging/current_cost���;�u��+       ��K	����A�-*

logging/current_cost���;����+       ��K	l��A�-*

logging/current_cost��;�87_+       ��K	t3��A�-*

logging/current_cost�~�;�!��+       ��K	�`��A�-*

logging/current_cost��;'�+       ��K	y���A�-*

logging/current_costnd�;q�
�+       ��K	���A�-*

logging/current_cost�3�;��|�+       ��K	/���A�-*

logging/current_costE��;�n2
+       ��K	~��A�-*

logging/current_cost,@�;�W#0+       ��K	�L��A�-*

logging/current_costՒ�;C_��+       ��K	pz��A�-*

logging/current_cost���;���+       ��K	v���A�-*

logging/current_cost0��;�F��+       ��K	����A�-*

logging/current_cost<��;ݘC/+       ��K	l��A�-*

logging/current_cost���;=_�+       ��K	�/��A�-*

logging/current_cost��;n�}�+       ��K	�^��A�-*

logging/current_cost��;H��J+       ��K	���A�-*

logging/current_cost$��;1fp+       ��K	R���A�-*

logging/current_cost0D�;�.?w+       ��K	n���A�-*

logging/current_costI�;F�
�+       ��K	� ��A�-*

logging/current_costA�;e�2+       ��K	ZD ��A�-*

logging/current_costn�;*q�+       ��K	.t ��A�-*

logging/current_cost�U�;��R+       ��K	/� ��A�-*

logging/current_cost�H�;�!>+       ��K	n� ��A�.*

logging/current_cost�	�;�n�!+       ��K	�� ��A�.*

logging/current_cost  �;nr��+       ��K	z/!��A�.*

logging/current_cost���;w��+       ��K	A_!��A�.*

logging/current_cost$��;�R"K+       ��K	ƌ!��A�.*

logging/current_cost h�;��J�+       ��K	��!��A�.*

logging/current_coste��;��l+       ��K	��!��A�.*

logging/current_cost�Q�;��N|+       ��K	X"��A�.*

logging/current_costU=�;���+       ��K	I"��A�.*

logging/current_cost�}�;�2��+       ��K	�v"��A�.*

logging/current_costr��;�ξ~+       ��K	��"��A�.*

logging/current_cost��;w�H+       ��K	f�"��A�.*

logging/current_cost��;�͸+       ��K	�#��A�.*

logging/current_cost��;����+       ��K	�.#��A�.*

logging/current_costU��;��W+       ��K	�`#��A�.*

logging/current_cost���;c�;�+       ��K	�#��A�.*

logging/current_cost^�;�"�+       ��K	R�#��A�.*

logging/current_costټ�;��d+       ��K	��#��A�.*

logging/current_costB�;,�x+       ��K	P$��A�.*

logging/current_cost�,�;�;��+       ��K	�K$��A�.*

logging/current_cost��;���+       ��K	Pz$��A�.*

logging/current_cost��;�E?	+       ��K	�$��A�.*

logging/current_costDC�;-:� +       ��K	f�$��A�.*

logging/current_cost��;[+       ��K	o%��A�.*

logging/current_costi��;��H�+       ��K	e3%��A�.*

logging/current_cost���;���+       ��K	�`%��A�.*

logging/current_cost�#�;(��+       ��K	�%��A�/*

logging/current_costb��;��+       ��K	q�%��A�/*

logging/current_cost��;9�:�+       ��K	.�%��A�/*

logging/current_cost��;�2�q+       ��K	�&��A�/*

logging/current_cost��;˾��+       ��K	M&��A�/*

logging/current_cost ��;j���+       ��K	�z&��A�/*

logging/current_cost��;!�y+       ��K	%�&��A�/*

logging/current_cost�8�;.;��+       ��K	��&��A�/*

logging/current_cost<��;��B�+       ��K	�'��A�/*

logging/current_cost��;B�Q+       ��K	�='��A�/*

logging/current_costǟ�;�LS�+       ��K	Zk'��A�/*

logging/current_costb�;�2�I+       ��K	��'��A�/*

logging/current_cost��;��1H+       ��K	o�'��A�/*

logging/current_cost�~�;Q��]+       ��K	%�'��A�/*

logging/current_cost��;6h��+       ��K	s%(��A�/*

logging/current_cost��;Gy_d+       ��K	�R(��A�/*

logging/current_cost0�;c��+       ��K	��(��A�/*

logging/current_costR��;�w|2+       ��K	8�(��A�/*

logging/current_cost�k�;bY�3+       ��K	�(��A�/*

logging/current_cost\/�;,Cq�+       ��K	I)��A�/*

logging/current_cost�|�;�1 �+       ��K	b6)��A�/*

logging/current_cost��;����+       ��K	c)��A�/*

logging/current_cost�~�;�/�+       ��K	{�)��A�/*

logging/current_cost���;�b'8+       ��K	=�)��A�/*

logging/current_cost���;M��+       ��K	��)��A�/*

logging/current_cost���;,h�+       ��K	%*��A�0*

logging/current_cost^�;�7��+       ��K	�K*��A�0*

logging/current_cost���;շp+       ��K	Ox*��A�0*

logging/current_cost�x�;���+       ��K	3�*��A�0*

logging/current_cost�1�;��FT+       ��K	�*��A�0*

logging/current_costG��;nY&+       ��K	Q+��A�0*

logging/current_cost9�;��~+       ��K	T0+��A�0*

logging/current_cost���;��mp+       ��K	g_+��A�0*

logging/current_cost[�;�+~	+       ��K	�+��A�0*

logging/current_cost�$�;���+       ��K	"�+��A�0*

logging/current_cost�F�;ȿH+       ��K	��+��A�0*

logging/current_costdi�;(��+       ��K	�,��A�0*

logging/current_cost��;q>�)+       ��K	rA,��A�0*

logging/current_cost���;����+       ��K	�o,��A�0*

logging/current_cost `�;(�t+       ��K	��,��A�0*

logging/current_cost>�;�&�5+       ��K	J�,��A�0*

logging/current_cost���;p�c~+       ��K	��,��A�0*

logging/current_cost'7�;�h<+       ��K	�'-��A�0*

logging/current_cost�;�;���W+       ��K	"U-��A�0*

logging/current_cost���;ο��+       ��K	��-��A�0*

logging/current_cost�'�;f��+       ��K	�-��A�0*

logging/current_costD��;��+       ��K	3�-��A�0*

logging/current_cost���;A{o�+       ��K	Y.��A�0*

logging/current_cost���;7��+       ��K	�A.��A�0*

logging/current_cost<,�;!��j+       ��K	oo.��A�0*

logging/current_cost�R�;=��~+       ��K	��.��A�0*

logging/current_costKO�;f��H+       ��K	��.��A�1*

logging/current_costGo�;	�+       ��K	R�.��A�1*

logging/current_cost�=�;�ّ�+       ��K	x'/��A�1*

logging/current_cost"G�;��#�+       ��K	�R/��A�1*

logging/current_coste��;�1�*+       ��K	�~/��A�1*

logging/current_costN�;��\�+       ��K	M�/��A�1*

logging/current_cost��;���+       ��K	��/��A�1*

logging/current_cost+��;;l�+       ��K	U0��A�1*

logging/current_cost�"�;5��d+       ��K	�;0��A�1*

logging/current_costR��;}?D>+       ��K	k0��A�1*

logging/current_cost���;<��(+       ��K	 �0��A�1*

logging/current_cost�;۹�+       ��K	��0��A�1*

logging/current_cost�(�;gBR�+       ��K	O�0��A�1*

logging/current_cost���;/�h+       ��K	a$1��A�1*

logging/current_costǾ�;�e�+       ��K	S1��A�1*

logging/current_cost��;��k�+       ��K	�1��A�1*

logging/current_cost���;�#�x+       ��K	��1��A�1*

logging/current_cost���;��+       ��K	��1��A�1*

logging/current_cost�8�;�H�+       ��K	2��A�1*

logging/current_costr��;͗Ԭ+       ��K	�=2��A�1*

logging/current_cost��;����+       ��K	5k2��A�1*

logging/current_cost��;�H�b+       ��K	��2��A�1*

logging/current_cost�(�;��zY+       ��K	*�2��A�1*

logging/current_costǟ�;�l��+       ��K	�2��A�1*

logging/current_cost��;'�M+       ��K	''3��A�1*

logging/current_cost���;	�ܼ+       ��K	pV3��A�2*

logging/current_costP��;�_ux+       ��K	��3��A�2*

logging/current_coste�;	��q+       ��K	g�3��A�2*

logging/current_cost	j�;��P+       ��K	d�3��A�2*

logging/current_cost2B�;Ns�)+       ��K	�4��A�2*

logging/current_cost�|�;شDi+       ��K	(;4��A�2*

logging/current_cost���;Sf�+       ��K	�i4��A�2*

logging/current_costD��;c	K�+       ��K	r�4��A�2*

logging/current_costk�;��\�+       ��K	F�4��A�2*

logging/current_cost�^�;|1��+       ��K	�4��A�2*

logging/current_cost)%�;�+�`+       ��K	�"5��A�2*

logging/current_costn�;u�C+       ��K	�N5��A�2*

logging/current_cost'J�;�ST+       ��K	J}5��A�2*

logging/current_cost���;��0g+       ��K	��5��A�2*

logging/current_costg�;��N�+       ��K	u�5��A�2*

logging/current_cost���;�Y<+       ��K	6��A�2*

logging/current_cost�;��.@+       ��K	^26��A�2*

logging/current_costN��;�;̢+       ��K	�^6��A�2*

logging/current_cost���;S�FN+       ��K	܌6��A�2*

logging/current_costM�;�#�+       ��K	��6��A�2*

logging/current_cost[��;٧��+       ��K	y�6��A�2*

logging/current_costRB�;����+       ��K	�7��A�2*

logging/current_cost	��;SW��+       ��K	6F7��A�2*

logging/current_cost�R�;�C�&+       ��K	t7��A�2*

logging/current_cost���;�5�+       ��K	
�7��A�2*

logging/current_cost�j�;�#?�+       ��K	�7��A�2*

logging/current_cost"��;���3+       ��K	v�7��A�3*

logging/current_costқ�; S�+       ��K	s+8��A�3*

logging/current_costU��;�݌�+       ��K	�X8��A�3*

logging/current_costU
�;���+       ��K	�8��A�3*

logging/current_cost}�;E8�V+       ��K	R�8��A�3*

logging/current_cost�#�;[W+       ��K	F�8��A�3*

logging/current_cost���;��@+       ��K	�9��A�3*

logging/current_costdg�;�9Á+       ��K	�>9��A�3*

logging/current_cost��;���E+       ��K	2l9��A�3*

logging/current_cost��;�;�M+       ��K	��9��A�3*

logging/current_costwl�;��;+       ��K	G�9��A�3*

logging/current_cost���;����+       ��K	$�9��A�3*

logging/current_cost�H�;�3ގ+       ��K	t#:��A�3*

logging/current_cost΃�;K�W+       ��K	0T:��A�3*

logging/current_costn>�;J��j+       ��K	܁:��A�3*

logging/current_costk�;#[�+       ��K	%�:��A�3*

logging/current_costDm�;���+       ��K	k�:��A�3*

logging/current_cost��;���+       ��K	�;��A�3*

logging/current_cost���;����+       ��K	�;;��A�3*

logging/current_cost��;$F�+       ��K	M;��A�3*

logging/current_cost�;�;c[�+       ��K	��;��A�3*

logging/current_cost���;z���+       ��K		<��A�3*

logging/current_cost��;���e+       ��K	�H<��A�3*

logging/current_cost�#�;��$�+       ��K	��<��A�3*

logging/current_cost���;� +       ��K	ؿ<��A�3*

logging/current_cost%��;�\�+       ��K	��<��A�3*

logging/current_cost̊�;.H�[+       ��K	�4=��A�4*

logging/current_cost���;2��+       ��K	g=��A�4*

logging/current_cost�Y�;�;z�+       ��K	�=��A�4*

logging/current_cost��;�C�x+       ��K	w�=��A�4*

logging/current_cost���;Ds5_+       ��K	�>��A�4*

logging/current_cost�h�;�$�+       ��K	6:>��A�4*

logging/current_cost��;�i�5+       ��K	�m>��A�4*

logging/current_cost9r�;���+       ��K	��>��A�4*

logging/current_cost`�;���+       ��K	�>��A�4*

logging/current_cost�U�;��z�+       ��K	�
?��A�4*

logging/current_cost���;�
�+       ��K	�=?��A�4*

logging/current_cost5�;I#�+       ��K	�q?��A�4*

logging/current_costފ�;�ʃ�+       ��K	��?��A�4*

logging/current_cost�;3¶�+       ��K	��?��A�4*

logging/current_cost��;!���+       ��K	[�?��A�4*

logging/current_cost�'�;�sZ$+       ��K	�*@��A�4*

logging/current_cost�/�;���+       ��K	~Y@��A�4*

logging/current_costE��;�u��+       ��K	4�@��A�4*

logging/current_cost<n�;�d �+       ��K	�@��A�4*

logging/current_cost$H�;:v�+       ��K	��@��A�4*

logging/current_cost��;? �+       ��K	cA��A�4*

logging/current_cost++�;�S�+       ��K	�HA��A�4*

logging/current_costng�;4��b+       ��K	�xA��A�4*

logging/current_costl��;��#:+       ��K	>�A��A�4*

logging/current_costy��;"��Y+       ��K	s�A��A�4*

logging/current_cost���;
���+       ��K	B��A�5*

logging/current_costO�;̌�+       ��K	�4B��A�5*

logging/current_cost5\�;��S+       ��K		kB��A�5*

logging/current_cost���;�H)+       ��K	s�B��A�5*

logging/current_cost���;+(+       ��K	��B��A�5*

logging/current_cost���;p�+       ��K	)�B��A�5*

logging/current_cost�O�;��+       ��K	d%C��A�5*

logging/current_cost^�;/�@�+       ��K	�TC��A�5*

logging/current_cost��;���+       ��K	(�C��A�5*

logging/current_cost���;�
��+       ��K	��C��A�5*

logging/current_costP��;9��N+       ��K	��C��A�5*

logging/current_cost���;�/�+       ��K	$D��A�5*

logging/current_cost���;/J��+       ��K	A:D��A�5*

logging/current_cost���;�+       ��K	,iD��A�5*

logging/current_costb �;��F�+       ��K	��D��A�5*

logging/current_cost��;閿�+       ��K	/�D��A�5*

logging/current_costk��;c<r+       ��K	�D��A�5*

logging/current_costՕ�;)O�\+       ��K	�E��A�5*

logging/current_costf�;u@+       ��K	�HE��A�5*

logging/current_cost���;��=�+       ��K	owE��A�5*

logging/current_cost}�;���R+       ��K	C�E��A�5*

logging/current_cost^y�;S��+       ��K	�E��A�5*

logging/current_costR��;O�eW+       ��K	� F��A�5*

logging/current_cost���;��n,+       ��K	�-F��A�5*

logging/current_cost���;4��+       ��K	!ZF��A�5*

logging/current_cost���;�@�0+       ��K	��F��A�5*

logging/current_cost��;D��P+       ��K	��F��A�6*

logging/current_costN��;�V�p+       ��K	b�F��A�6*

logging/current_cost�L�;���;+       ��K	G��A�6*

logging/current_coste��;s��+       ��K	YDG��A�6*

logging/current_cost;��;��
+       ��K	�qG��A�6*

logging/current_cost���;}�0'+       ��K	��G��A�6*

logging/current_cost���;����+       ��K	L�G��A�6*

logging/current_cost�G�;ԋ+       ��K	�)H��A�6*

logging/current_cost<��;��h+       ��K	5`H��A�6*

logging/current_cost�M�;օ�+       ��K	��H��A�6*

logging/current_cost���;�Ѯ0+       ��K	��H��A�6*

logging/current_cost���;z��+       ��K	1I��A�6*

logging/current_cost�F�;Ea�&+       ��K		TI��A�6*

logging/current_cost�
�;	���+       ��K	0�I��A�6*

logging/current_cost,t�;�L+       ��K	��I��A�6*

logging/current_cost���;q�>�+       ��K	#J��A�6*

logging/current_cost��;���N+       ��K	YJ��A�6*

logging/current_cost���;U8J�+       ��K	Q�J��A�6*

logging/current_cost^�;�h��+       ��K	��J��A�6*

logging/current_cost�G�;뛷&+       ��K	�K��A�6*

logging/current_cost���;Cc�W+       ��K	�KK��A�6*

logging/current_cost���;w;��+       ��K	��K��A�6*

logging/current_cost;A�;��Ȑ+       ��K	��K��A�6*

logging/current_costy��;�ya+       ��K	�L��A�6*

logging/current_cost�z�;���+       ��K	�8L��A�6*

logging/current_cost�;|d�+       ��K	&yL��A�7*

logging/current_costn!�;7��+       ��K	O�L��A�7*

logging/current_cost~�;�9�+       ��K	:�L��A�7*

logging/current_cost��;y�v�+       ��K	jM��A�7*

logging/current_cost{6�;�@��+       ��K	�JM��A�7*

logging/current_cost+��;�J�+       ��K	wM��A�7*

logging/current_costե�;�۱P+       ��K	δM��A�7*

logging/current_cost���;��?�+       ��K	��M��A�7*

logging/current_cost�w�;�VY+       ��K	�N��A�7*

logging/current_cost�%�;���Q+       ��K	^N��A�7*

logging/current_cost���;�-�h+       ��K	��N��A�7*

logging/current_cost@��;tnx�+       ��K	��N��A�7*

logging/current_cost���;Z�+       ��K	=O��A�7*

logging/current_cost���;6!��+       ��K	�\O��A�7*

logging/current_cost�v�;���V+       ��K	r�O��A�7*

logging/current_cost'��;���+       ��K	��O��A�7*

logging/current_cost���;�Q9�+       ��K	��O��A�7*

logging/current_cost�e�;O���+       ��K	L&P��A�7*

logging/current_cost�;?O{K+       ��K	�XP��A�7*

logging/current_cost���;�Ƥ+       ��K	d�P��A�7*

logging/current_cost���;��+       ��K	'�P��A�7*

logging/current_cost|�;v��7+       ��K	TQ��A�7*

logging/current_cost�v�;�Cnb+       ��K		6Q��A�7*

logging/current_cost���;��T+       ��K	6kQ��A�7*

logging/current_cost"��;N�"h+       ��K	��Q��A�7*

logging/current_cost���;%��+       ��K	��Q��A�7*

logging/current_cost4�;��}w+       ��K	'R��A�8*

logging/current_cost���;䌔�+       ��K	�7R��A�8*

logging/current_cost�0�;��Z+       ��K	ufR��A�8*

logging/current_cost`��;��_�+       ��K	��R��A�8*

logging/current_cost���;�V +       ��K	b�R��A�8*

logging/current_cost�f�;�u8+       ��K	>S��A�8*

logging/current_cost���;1�.&+       ��K	3S��A�8*

logging/current_cost��;���<+       ��K	OfS��A�8*

logging/current_costk6�;{�mk+       ��K	u�S��A�8*

logging/current_cost5\�;�?�+       ��K	Q�S��A�8*

logging/current_cost'�;x) b+       ��K	mT��A�8*

logging/current_cost�I�;~@�+       ��K	�@T��A�8*

logging/current_cost�X�;p��P+       ��K	�rT��A�8*

logging/current_cost<s�;�0N�+       ��K	��T��A�8*

logging/current_cost.��;�+       ��K	^
U��A�8*

logging/current_cost@�;GPe�+       ��K	>U��A�8*

logging/current_cost���;��"�+       ��K	�vU��A�8*

logging/current_cost��;�`��+       ��K	��U��A�8*

logging/current_cost�B�;�W+       ��K	:�U��A�8*

logging/current_cost��;	��+       ��K	�V��A�8*

logging/current_cost L�;���+       ��K	�SV��A�8*

logging/current_cost��;
�em+       ��K	�V��A�8*

logging/current_cost0��;�U��+       ��K	8�V��A�8*

logging/current_cost�\�;aĬ�+       ��K	QW��A�8*

logging/current_cost� �; ���+       ��K	bAW��A�8*

logging/current_cost�y�;��F+       ��K	�qW��A�8*

logging/current_cost�[�;�L�+       ��K	w�W��A�9*

logging/current_costr��;��P++       ��K	��W��A�9*

logging/current_cost���;F��<+       ��K	JX��A�9*

logging/current_cost���;�-�+       ��K	�IX��A�9*

logging/current_costT��;��l�+       ��K	�|X��A�9*

logging/current_costr��;o�Gj+       ��K	��X��A�9*

logging/current_cost��;�e�+       ��K	��X��A�9*

logging/current_costd#�;V�+       ��K	k2Y��A�9*

logging/current_cost�j�;C�۬+       ��K	�cY��A�9*

logging/current_cost���;E+       ��K	��Y��A�9*

logging/current_cost	�;����+       ��K	��Y��A�9*

logging/current_cost���;�	�+       ��K	"Z��A�9*

logging/current_costd��;��U+       ��K	}9Z��A�9*

logging/current_cost���;��� +       ��K	IiZ��A�9*

logging/current_cost�!�;wZ-�+       ��K	�Z��A�9*

logging/current_costEe�;@�x+       ��K	-�Z��A�9*

logging/current_cost ��;�uGi+       ��K	R�Z��A�9*

logging/current_cost��;6 �+       ��K	�/[��A�9*

logging/current_costd|�;*�RP+       ��K	�b[��A�9*

logging/current_costG��;p���+       ��K	+�[��A�9*

logging/current_cost;��;��+       ��K	S�[��A�9*

logging/current_cost���;�Fʇ+       ��K	f\��A�9*

logging/current_cost��;J��+       ��K	�8\��A�9*

logging/current_cost��;6+       ��K	�i\��A�9*

logging/current_cost$0�;*��+       ��K	J�\��A�9*

logging/current_cost���;�Ay+       ��K	��\��A�:*

logging/current_cost���;���m+       ��K	3]��A�:*

logging/current_cost�&�;[c_�+       ��K	�B]��A�:*

logging/current_costd?�;�(+       ��K	�|]��A�:*

logging/current_costYJ�;�`L,+       ��K	 �]��A�:*

logging/current_cost~/�;P��+       ��K	X�]��A�:*

logging/current_cost$��;Q��1+       ��K	 ^��A�:*

logging/current_costy_�;N�Z+       ��K	�J^��A�:*

logging/current_cost��;��p�+       ��K	T|^��A�:*

logging/current_cost��;��/-+       ��K	ů^��A�:*

logging/current_cost��;���+       ��K	��^��A�:*

logging/current_cost�b�;��i+       ��K	�_��A�:*

logging/current_cost���;I+"+       ��K	�=_��A�:*

logging/current_cost� �;�A�+       ��K	3m_��A�:*

logging/current_cost�2�;^�׶+       ��K	x�_��A�:*

logging/current_cost�5�;h�4+       ��K	��_��A�:*

logging/current_costۗ�;�U�+       ��K	��_��A�:*

logging/current_cost���;�뫐+       ��K	%`��A�:*

logging/current_cost+B�;b�,�+       ��K	�U`��A�:*

logging/current_cost���;2��?+       ��K	��`��A�:*

logging/current_cost���;����+       ��K	��`��A�:*

logging/current_cost�7�;'
À+       ��K	��`��A�:*

logging/current_costG�;.�yi+       ��K	#	a��A�:*

logging/current_cost�I�;�K�)+       ��K	.8a��A�:*

logging/current_cost[}�;�.��+       ��K	ea��A�:*

logging/current_costۋ�; ̃+       ��K	Y�a��A�:*

logging/current_costg�;�b%�+       ��K	�a��A�;*

logging/current_cost+�;��L+       ��K	]�a��A�;*

logging/current_costM�;��7F+       ��K	�b��A�;*

logging/current_cost���;ifc�+       ��K	Kb��A�;*

logging/current_cost%*�;�K�+       ��K	|}b��A�;*

logging/current_costwd�;龻�+       ��K	�b��A�;*

logging/current_cost���;|��r+       ��K	8�b��A�;*

logging/current_cost���;P,ݎ+       ��K	�	c��A�;*

logging/current_cost�m�;�FN+       ��K	�7c��A�;*

logging/current_cost[��;��� +       ��K	jgc��A�;*

logging/current_cost��;����+       ��K	��c��A�;*

logging/current_cost�@�;��|A+       ��K	5�c��A�;*

logging/current_cost\.�;��~�+       ��K	�c��A�;*

logging/current_cost5��;7�,�+       ��K	�d��A�;*

logging/current_costy��;���c+       ��K	�Md��A�;*

logging/current_cost4��;`/2�+       ��K	}�d��A�;*

logging/current_cost;��;���?+       ��K	��d��A�;*

logging/current_cost���;�C�L+       ��K	��d��A�;*

logging/current_cost���;���+       ��K	'e��A�;*

logging/current_cost���;{R�Y+       ��K	�;e��A�;*

logging/current_costN3�;KpY+       ��K	�le��A�;*

logging/current_cost�o�;�J}#+       ��K	�e��A�;*

logging/current_cost��;�3:_+       ��K	A�e��A�;*

logging/current_cost &�;���+       ��K	��e��A�;*

logging/current_cost|��;�P�+       ��K	,f��A�;*

logging/current_cost�W�;	�<�+       ��K	�Zf��A�<*

logging/current_cost�;M���+       ��K	w�f��A�<*

logging/current_costж�;�N�_+       ��K	��f��A�<*

logging/current_cost ��;9�m�+       ��K	h�f��A�<*

logging/current_cost��;m�1h+       ��K	�g��A�<*

logging/current_cost��;��+�+       ��K	�Kg��A�<*

logging/current_cost�	�;T52+       ��K	({g��A�<*

logging/current_cost���;݄��+       ��K	=�g��A�<*

logging/current_cost���;mԾ�+       ��K	��g��A�<*

logging/current_cost�3�;�fGC+       ��K	�	h��A�<*

logging/current_cost�E�;�M�+       ��K	�:h��A�<*

logging/current_cost���;��+       ��K	Fgh��A�<*

logging/current_cost\0�;�٘w+       ��K	Öh��A�<*

logging/current_costu�;��n~+       ��K	�h��A�<*

logging/current_cost���;u�P�+       ��K	&�h��A�<*

logging/current_costP.�;!��+       ��K	�#i��A�<*

logging/current_cost��;;��+       ��K	WQi��A�<*

logging/current_cost�2�;(��+       ��K	1~i��A�<*

logging/current_cost"��;_�+       ��K	^�i��A�<*

logging/current_cost��;�R�O+       ��K	x�i��A�<*

logging/current_cost2d�;3M�+       ��K	�j��A�<*

logging/current_costҵ�;�M`+       ��K	4j��A�<*

logging/current_costq�;3�J�+       ��K	#_j��A�<*

logging/current_cost~�;�s�+       ��K	[�j��A�<*

logging/current_cost#�;`s��+       ��K	Թj��A�<*

logging/current_costy��;7Xh+       ��K	@�j��A�<*

logging/current_cost���;z�+       ��K	k��A�=*

logging/current_cost�W�;S�a+       ��K	�@k��A�=*

logging/current_cost`R�;�n�+       ��K	�mk��A�=*

logging/current_cost���;�*S+       ��K	��k��A�=*

logging/current_cost.��;��'�+       ��K	��k��A�=*

logging/current_cost0��;A_��+       ��K	T�k��A�=*

logging/current_cost�t�;�P4r+       ��K	�#l��A�=*

logging/current_cost���;2��+       ��K	OQl��A�=*

logging/current_cost�,�;쭆�+       ��K	�l��A�=*

logging/current_cost���;0:=�+       ��K	��l��A�=*

logging/current_cost�$�;ŵć+       ��K	D�l��A�=*

logging/current_cost���;��h�+       ��K	�m��A�=*

logging/current_cost��;���+       ��K	O4m��A�=*

logging/current_costǛ�;���+       ��K	�bm��A�=*

logging/current_cost <�;~q��+       ��K	)�m��A�=*

logging/current_cost�V�;�~+       ��K	�m��A�=*

logging/current_cost���;�Cx�+       ��K	��m��A�=*

logging/current_cost<4�;����+       ��K	(n��A�=*

logging/current_cost��;+5+       ��K	�Bn��A�=*

logging/current_cost���;2#�+       ��K	�yn��A�=*

logging/current_costbr�;���,+       ��K	��n��A�=*

logging/current_cost�a�;I�	�+       ��K	��n��A�=*

logging/current_cost�k�;���+       ��K	o��A�=*

logging/current_cost>��;��+       ��K	E4o��A�=*

logging/current_cost׶�;`O�+       ��K	�do��A�=*

logging/current_cost���;��,++       ��K	�o��A�=*

logging/current_cost�'�;�}{+       ��K	��o��A�>*

logging/current_costt�;�p�+       ��K	��o��A�>*

logging/current_costպ�;^�3+       ��K	� p��A�>*

logging/current_cost�_�;6�;3+       ��K	�Pp��A�>*

logging/current_cost�#�;��++       ��K	p��A�>*

logging/current_costЖ�;z�$�+       ��K	B�p��A�>*

logging/current_cost���;�
V�+       ��K	�p��A�>*

logging/current_cost9^�;>D�7+       ��K	�
q��A�>*

logging/current_costE��;���+       ��K	:q��A�>*

logging/current_cost���;<��+       ��K	�gq��A�>*

logging/current_cost���;�y��+       ��K	*�q��A�>*

logging/current_costJ�;��ɇ+       ��K	��q��A�>*

logging/current_cost �;�T�d+       ��K	T�q��A�>*

logging/current_cost���;��=+       ��K	"r��A�>*

logging/current_costP��;o���+       ��K	`Qr��A�>*

logging/current_cost���;��+�+       ��K	�r��A�>*

logging/current_cost���;
C�+       ��K	R�r��A�>*

logging/current_cost���;�Uv+       ��K	��r��A�>*

logging/current_costҩ�;��x+       ��K	�s��A�>*

logging/current_cost0��;�kP;+       ��K	|<s��A�>*

logging/current_cost���;S��&+       ��K	�js��A�>*

logging/current_cost���;�`�+       ��K	�s��A�>*

logging/current_cost��;{9�+       ��K	R�s��A�>*

logging/current_costǟ�;�F�+       ��K	��s��A�>*

logging/current_cost���;����+       ��K	�$t��A�>*

logging/current_cost�;���+       ��K	�Rt��A�?*

logging/current_costL�;�r�H+       ��K	a�t��A�?*

logging/current_cost҂�;��;�+       ��K	��t��A�?*

logging/current_cost@g�;�s��+       ��K	�t��A�?*

logging/current_cost'��;��+       ��K	�u��A�?*

logging/current_cost���;�`a+       ��K	E>u��A�?*

logging/current_cost\��;'2}+       ��K	enu��A�?*

logging/current_cost���;�2�+       ��K	i�u��A�?*

logging/current_cost��;�-߂+       ��K	/�u��A�?*

logging/current_cost^��;�٘�+       ��K	s�u��A�?*

logging/current_costE��;���+       ��K	M(v��A�?*

logging/current_cost��;_M�3+       ��K	�Uv��A�?*

logging/current_cost<u�;,DO�+       ��K	��v��A�?*

logging/current_cost̆�;�H�+       ��K	ٰv��A�?*

logging/current_cost���;�A�+       ��K	��v��A�?*

logging/current_cost���;s�+       ��K	�w��A�?*

logging/current_cost��;�Y��+       ��K	�9w��A�?*

logging/current_cost�;]��+       ��K	�ew��A�?*

logging/current_cost[k�;׊�+       ��K	�w��A�?*

logging/current_costa�;���+       ��K	��w��A�?*

logging/current_cost��;���l+       ��K	��w��A�?*

logging/current_cost���;�w_ +       ��K	�x��A�?*

logging/current_cost��;�9�y+       ��K	hGx��A�?*

logging/current_cost2T�;젒+       ��K	�vx��A�?*

logging/current_cost$��;@_�w+       ��K	g�x��A�?*

logging/current_cost���;�:�+       ��K	��x��A�?*

logging/current_costէ�;�v�z+       ��K	=y��A�@*

logging/current_costP��;�!~k+       ��K	(0y��A�@*

logging/current_costR!�;cɹ|+       ��K	�\y��A�@*

logging/current_cost�	�;�4�j+       ��K	��y��A�@*

logging/current_cost ��;�4;�+       ��K	��y��A�@*

logging/current_costs�;K+�)+       ��K	��y��A�@*

logging/current_costD��;����+       ��K	�z��A�@*

logging/current_cost�;k���+       ��K	�Ez��A�@*

logging/current_cost9a�;��K+       ��K	psz��A�@*

logging/current_costG&�;�8�+       ��K	ڟz��A�@*

logging/current_costN��;����+       ��K	��z��A�@*

logging/current_cost��;����+       ��K	��z��A�@*

logging/current_costd��;���#+       ��K	/({��A�@*

logging/current_cost��;hI�+       ��K	�S{��A�@*

logging/current_cost`��;���+       ��K	��{��A�@*

logging/current_cost5O�;�Ą+       ��K	��{��A�@*

logging/current_cost�W�;%��+       ��K	�|��A�@*

logging/current_costw��;wA=j+       ��K	�K|��A�@*

logging/current_costh�;�n��+       ��K	<�|��A�@*

logging/current_cost�O�;(J��+       ��K	}��A�@*

logging/current_cost�w�;�+       ��K	�8}��A�@*

logging/current_cost[1�;���N+       ��K	0r}��A�@*

logging/current_cost�X�;2�D+       ��K	.�}��A�@*

logging/current_cost���;%x+       ��K	t�}��A�@*

logging/current_costwZ�;U�X�+       ��K	�~��A�@*

logging/current_cost�;�;j�d+       ��K	W~��A�A*

logging/current_costu��;��+       ��K	x�~��A�A*

logging/current_cost<U�;��`4+       ��K	ٺ~��A�A*

logging/current_cost��;��+       ��K	��~��A�A*

logging/current_cost���;;=V+       ��K	���A�A*

logging/current_cost��;%��+       ��K	�M��A�A*

logging/current_cost@�;N;_�+       ��K	�~��A�A*

logging/current_cost\��;b�+       ��K	k���A�A*

logging/current_cost'��;(��T+       ��K	����A�A*

logging/current_cost� �;����+       ��K	����A�A*

logging/current_costW�;�~n+       ��K	.:���A�A*

logging/current_cost��;y��+       ��K	;i���A�A*

logging/current_costE��;�Fz�+       ��K	�����A�A*

logging/current_costI�;����+       ��K	�ɀ��A�A*

logging/current_cost��;YH@�+       ��K	:����A�A*

logging/current_cost9��;3Ы+       ��K	,���A�A*

logging/current_cost���;�*+       ��K	�t���A�A*

logging/current_costr��;٘1�+       ��K	Ը���A�A*

logging/current_costU��;�q+       ��K	�����A�A*

logging/current_costg��;���\+       ��K	{;���A�A*

logging/current_cost'5�;V��+       ��K	�����A�A*

logging/current_costى�;~�{z+       ��K	�Ƃ��A�A*

logging/current_cost��;�0H +       ��K	�����A�A*

logging/current_cost�%�;M�d�+       ��K	??���A�A*

logging/current_costD��;��8+       ��K	l|���A�A*

logging/current_cost���;tXll+       ��K	I����A�A*

logging/current_costU7�;��D�+       ��K	���A�B*

logging/current_cost��;��+       ��K	�(���A�B*

logging/current_costċ�;�f�+       ��K	�\���A�B*

logging/current_cost��;�by�+       ��K	�����A�B*

logging/current_cost�m�;�b3w+       ��K	�Ä��A�B*

logging/current_cost���;Y��+       ��K	�����A�B*

logging/current_cost;�;Ϗ�\+       ��K	�'���A�B*

logging/current_cost�O�;y��Y+       ��K	�[���A�B*

logging/current_cost���;���+       ��K	�����A�B*

logging/current_cost@y�;���+       ��K	���A�B*

logging/current_cost�[�;(�#�+       ��K	����A�B*

logging/current_cost���;�Fio+       ��K	"���A�B*

logging/current_costI��;���+       ��K	�N���A�B*

logging/current_cost�7�;8���+       ��K	�}���A�B*

logging/current_costUt�;���+       ��K	�����A�B*

logging/current_cost���;�xw�+       ��K	�چ��A�B*

logging/current_cost���;_D�-+       ��K	k
���A�B*

logging/current_costī�;!�ٶ+       ��K	�7���A�B*

logging/current_costn
�;�+       ��K	6f���A�B*

logging/current_cost�<�;@�Z�+       ��K	ȓ���A�B*

logging/current_cost ��;2�8-+       ��K	+Ç��A�B*

logging/current_cost4r�;pm�+       ��K	�����A�B*

logging/current_cost^/�;D}8+       ��K	k4���A�B*

logging/current_costt��;g�Q�+       ��K	�b���A�B*

logging/current_cost�3�;����+       ��K	!����A�B*

logging/current_cost�,�;]�=&+       ��K	nǈ��A�B*

logging/current_cost.��;�l}+       ��K	�����A�C*

logging/current_cost'��;�%�+       ��K	�(���A�C*

logging/current_cost�K�;��+       ��K	TU���A�C*

logging/current_cost~S�;�Q��+       ��K	����A�C*

logging/current_costg�;5e�+       ��K	)����A�C*

logging/current_cost�;�k��+       ��K	�����A�C*

logging/current_cost���;w�!�+       ��K	}���A�C*

logging/current_cost���;z,:+       ��K	gR���A�C*

logging/current_cost���;�ze�+       ��K	����A�C*

logging/current_cost���;)�A�+       ��K	����A�C*

logging/current_cost��;K��+       ��K	Oފ��A�C*

logging/current_costN��;q��7+       ��K	M���A�C*

logging/current_costR��;��&0+       ��K	;���A�C*

logging/current_cost� �;�'�+       ��K	�g���A�C*

logging/current_cost�/�;X���+       ��K	�����A�C*

logging/current_cost���;O9~�+       ��K	�ɋ��A�C*

logging/current_cost+��;�y
+       ��K	�����A�C*

logging/current_costLS�;�*p�+       ��K	�&���A�C*

logging/current_cost���;��+       ��K	�W���A�C*

logging/current_cost�y�;�A.+       ��K	����A�C*

logging/current_costˠ�;8��1+       ��K	���A�C*

logging/current_cost���;ܓ��+       ��K	�����A�C*

logging/current_costtQ�;�^G+       ��K	k���A�C*

logging/current_costE�;�,��+       ��K	�C���A�C*

logging/current_cost���;�9{�+       ��K	�p���A�C*

logging/current_cost\�;&9/+       ��K	
����A�D*

logging/current_cost���;F�'+       ��K	�̍��A�D*

logging/current_costP��;[{�+       ��K	C����A�D*

logging/current_costM�;��X+       ��K	1*���A�D*

logging/current_costX�;�6*x+       ��K	�d���A�D*

logging/current_costE��;e�+       ��K	����A�D*

logging/current_cost�v�;�5[�+       ��K	�����A�D*

logging/current_cost,��;ҰxA+       ��K	[���A�D*

logging/current_cost�X�;��+       ��K	� ���A�D*

logging/current_costD.�;����+       ��K	R���A�D*

logging/current_cost5��;�YPH+       ��K	Y����A�D*

logging/current_costL��;�O7�+       ��K	Ԯ���A�D*

logging/current_cost���;jPc�+       ��K	�ۏ��A�D*

logging/current_costD��;�J�3+       ��K	����A�D*

logging/current_cost��;>��+       ��K	T;���A�D*

logging/current_cost 
�;v�Z+       ��K	�i���A�D*

logging/current_cost���;��"+       ��K	�����A�D*

logging/current_cost���;�.��+       ��K	QƐ��A�D*

logging/current_cost�&�;� +       ��K	�����A�D*

logging/current_cost k�;���B+       ��K	$���A�D*

logging/current_cost���;�54+       ��K	�Q���A�D*

logging/current_cost,h�;�g'�+       ��K	M����A�D*

logging/current_cost���;͔�+       ��K	�����A�D*

logging/current_cost>[�;M���+       ��K	rؑ��A�D*

logging/current_cost9�;c�+       ��K	����A�D*

logging/current_cost9��;>�L+       ��K	B3���A�D*

logging/current_cost5��;�"~�+       ��K	�_���A�E*

logging/current_cost�;��IU+       ��K	/����A�E*

logging/current_cost�I�;�~�T+       ��K	���A�E*

logging/current_cost��;�'H
+       ��K	����A�E*

logging/current_cost���;���+       ��K	����A�E*

logging/current_costBb�;~��+       ��K	=C���A�E*

logging/current_cost�3�;�U��+       ��K	�s���A�E*

logging/current_cost.�;3��+       ��K	�����A�E*

logging/current_cost�M�;1Q+       ��K	�Γ��A�E*

logging/current_cost�(�;0�>/+       ��K	� ���A�E*

logging/current_cost�t�;'ŊP+       ��K	�/���A�E*

logging/current_cost|�;5/�+       ��K	�]���A�E*

logging/current_costD��;8�|=+       ��K	���A�E*

logging/current_costg4�;-�6:+       ��K	�����A�E*

logging/current_cost�A�;�� y+       ��K	*���A�E*

logging/current_cost ��;���+       ��K	����A�E*

logging/current_cost9D�;����+       ��K	6B���A�E*

logging/current_costK��;����+       ��K	8q���A�E*

logging/current_cost���;rN�+       ��K	�����A�E*

logging/current_costW��;�2�+       ��K	�͕��A�E*

logging/current_cost�;��"+       ��K	�����A�E*

logging/current_cost�l�;���l+       ��K	�&���A�E*

logging/current_cost�C�;o��
+       ��K	�S���A�E*

logging/current_cost~N�;�S�W+       ��K	����A�E*

logging/current_cost'��;7u�+       ��K	0����A�E*

logging/current_costrx�;����+       ��K	ݖ��A�F*

logging/current_costkv�;M3�=+       ��K	%
���A�F*

logging/current_cost��;�`͵+       ��K	)<���A�F*

logging/current_costp��;���?+       ��K	Qq���A�F*

logging/current_cost��;���+       ��K	����A�F*

logging/current_cost+�;��+       ��K	�͗��A�F*

logging/current_cost@��;:{-�+       ��K	,����A�F*

logging/current_cost�!�;jx�%+       ��K	a.���A�F*

logging/current_costN��;�bqu+       ��K	�Z���A�F*

logging/current_cost�^�;�
�G+       ��K	�����A�F*

logging/current_cost���;'�3+       ��K	D����A�F*

logging/current_costnh�;k��	+       ��K	����A�F*

logging/current_costY��;���+       ��K	C���A�F*

logging/current_cost���;�
��+       ��K	FE���A�F*

logging/current_cost^��;�P�N+       ��K	Hv���A�F*

logging/current_cost��;�Zh�+       ��K	�����A�F*

logging/current_cost$�;���+       ��K	|֙��A�F*

logging/current_costU$�;k7�0+       ��K	-���A�F*

logging/current_costp��;L�$+       ��K	�4���A�F*

logging/current_cost��;�q+       ��K	�e���A�F*

logging/current_costܭ�;�k��+       ��K	�����A�F*

logging/current_costY��;%��+       ��K	nĚ��A�F*

logging/current_costUv�;z��Y+       ��K	a����A�F*

logging/current_cost�]�;���~+       ��K	3*���A�F*

logging/current_costLS�; ��J+       ��K	�X���A�F*

logging/current_cost�@�;Ni�T+       ��K	7����A�F*

logging/current_cost+N�;v�>
+       ��K	䳛��A�G*

logging/current_cost`r�;����+       ��K	����A�G*

logging/current_cost�Z�;��Z+       ��K	����A�G*

logging/current_costU��;O)�-+       ��K	�?���A�G*

logging/current_cost 
�;e�=�+       ��K	�l���A�G*

logging/current_cost+�;���g+       ��K	t����A�G*

logging/current_cost9�;!+       ��K	�ɜ��A�G*

logging/current_costb�;:j]+       ��K	�����A�G*

logging/current_cost���;*�NJ+       ��K	�(���A�G*

logging/current_costy��;��W�+       ��K	�W���A�G*

logging/current_cost��;�I�+       ��K	�����A�G*

logging/current_cost$��;��+�+       ��K	'����A�G*

logging/current_cost���;S��+       ��K	�ޝ��A�G*

logging/current_cost��;i���+       ��K	B���A�G*

logging/current_costD}�;��{+       ��K	�9���A�G*

logging/current_cost�s�;����+       ��K	�h���A�G*

logging/current_cost�}�;��^�+       ��K	[����A�G*

logging/current_cost��;A:�i+       ��K	Þ��A�G*

logging/current_cost�b�;D��X+       ��K	����A�G*

logging/current_cost���;m�:+       ��K	����A�G*

logging/current_costlp�;u ��+       ��K	AL���A�G*

logging/current_cost��;��#�+       ��K	@{���A�G*

logging/current_costky�;&��3+       ��K	Ԫ���A�G*

logging/current_cost���;T� G+       ��K	<ן��A�G*

logging/current_cost�p�;Q���+       ��K	���A�G*

logging/current_cost���;��g�+       ��K	D3���A�G*

logging/current_cost�r�;���n+       ��K	c���A�H*

logging/current_cost���;T4W�+       ��K	�����A�H*

logging/current_costnr�;���+       ��K	�����A�H*

logging/current_cost9��;7�N�+       ��K	�����A�H*

logging/current_cost��;�)�O+       ��K	����A�H*

logging/current_costn��;��M�+       ��K	pG���A�H*

logging/current_cost���;��+       ��K	Eu���A�H*

logging/current_cost��;c	��+       ��K	-����A�H*

logging/current_costJ�;��]�+       ��K	%ѡ��A�H*

logging/current_cost27�;w6� +       ��K	����A�H*

logging/current_costIx�;Ga�+       ��K	/���A�H*

logging/current_cost<9�;�E�B+       ��K	�\���A�H*

logging/current_costK��;,���+       ��K	[����A�H*

logging/current_cost���;@�Q+       ��K	����A�H*

logging/current_cost��;Jn�+       ��K	����A�H*

logging/current_cost$@�;_t�P+       ��K	����A�H*

logging/current_cost N�;9��+       ��K	hE���A�H*

logging/current_costRK�;Lϫn+       ��K	�s���A�H*

logging/current_cost���;��b+       ��K	����A�H*

logging/current_cost޽�;����+       ��K	�ϣ��A�H*

logging/current_cost�g�;�6�6+       ��K	d����A�H*

logging/current_cost���;�Z�2+       ��K	�+���A�H*

logging/current_cost���;�Z��+       ��K	M`���A�H*

logging/current_costGf�;���E+       ��K	܏���A�H*

logging/current_costΐ�;����+       ��K	����A�H*

logging/current_cost5�;_7�+       ��K	
���A�I*

logging/current_cost�B�;O.ە+       ��K	���A�I*

logging/current_costǾ�;u���+       ��K	DK���A�I*

logging/current_cost<3�;z �+       ��K	�{���A�I*

logging/current_costu!�;j�P+       ��K	����A�I*

logging/current_costwS�; �֔+       ��K	�٥��A�I*

logging/current_cost��;�	��+       ��K	����A�I*

logging/current_cost���;o\:�+       ��K	�9���A�I*

logging/current_cost,��;��g+       ��K	�g���A�I*

logging/current_cost���;��+       ��K	�����A�I*

logging/current_cost���;�/�+       ��K	�ͦ��A�I*

logging/current_cost%��;����+       ��K	�����A�I*

logging/current_cost.��;u+�+       ��K	�0���A�I*

logging/current_cost�p�;��+       ��K	�`���A�I*

logging/current_costB��;���+       ��K	ɑ���A�I*

logging/current_cost�|�;P�j�+       ��K	ڽ���A�I*

logging/current_cost�;���+       ��K	9���A�I*

logging/current_cost$��;��FH+       ��K	����A�I*

logging/current_cost+�;:��+       ��K	�K���A�I*

logging/current_cost"��;b4��+       ��K	�z���A�I*

logging/current_cost��;��h�+       ��K	a����A�I*

logging/current_cost�<�;���+       ��K	Pب��A�I*

logging/current_cost�[�;��/+       ��K	0���A�I*

logging/current_coste��;�h��+       ��K	&7���A�I*

logging/current_costI~�;��Z+       ��K	�f���A�I*

logging/current_cost���;l9+       ��K	����A�I*

logging/current_cost>q�;�q��+       ��K	6ϩ��A�J*

logging/current_cost\�;]kV�+       ��K	����A�J*

logging/current_costw�;��N+       ��K	�5���A�J*

logging/current_cost��;���+       ��K	cm���A�J*

logging/current_cost��;�fV�+       ��K	����A�J*

logging/current_cost���;aʓ+       ��K	�ɪ��A�J*

logging/current_costk�;6#�+       ��K	i����A�J*

logging/current_cost�d�;�´5+       ��K	�%���A�J*

logging/current_cost�n�;�b�+       ��K	mS���A�J*

logging/current_cost<?�;�<;�+       ��K	�����A�J*

logging/current_cost"��;���+       ��K	ج���A�J*

logging/current_cost�8�;��+       ��K	�۫��A�J*

logging/current_cost�+�;O4A;+       ��K	W
���A�J*

logging/current_cost�?�;9k��+       ��K	�5���A�J*

logging/current_cost9f�;�C �+       ��K	�b���A�J*

logging/current_costI�;@|�+       ��K	ᒬ��A�J*

logging/current_cost"��;~ԏ'+       ��K	+����A�J*

logging/current_cost���;� +       ��K	����A�J*

logging/current_cost�H�;*#��+       ��K	����A�J*

logging/current_costR��;���+       ��K	/H���A�J*

logging/current_cost��;�8�+       ��K	�u���A�J*

logging/current_costd��;9X�+       ��K	k����A�J*

logging/current_cost�?�;V+       ��K	�խ��A�J*

logging/current_cost�I�;���+       ��K	����A�J*

logging/current_cost,P�;5k��+       ��K	�3���A�J*

logging/current_costEq�;����+       ��K	8b���A�K*

logging/current_cost���;�sv+       ��K	�����A�K*

logging/current_costK�;����+       ��K	Ͽ���A�K*

logging/current_cost��;��+       ��K	�����A�K*

logging/current_costŉ�;ʀ4�+       ��K	����A�K*

logging/current_cost���;�N��+       ��K	�H���A�K*

logging/current_costY��;z)q+       ��K	�v���A�K*

logging/current_costu�;�~>�+       ��K	4����A�K*

logging/current_cost���;¬�+       ��K	�ϯ��A�K*

logging/current_cost���;	B��+       ��K	�����A�K*

logging/current_cost���;�nw+       ��K	�,���A�K*

logging/current_cost���;�xD+       ��K	�Y���A�K*

logging/current_cost���;~�*�+       ��K	慰��A�K*

logging/current_cost�:�;-��+       ��K	�����A�K*

logging/current_cost>��;����+       ��K	u���A�K*

logging/current_cost���;�lL+       ��K	����A�K*

logging/current_cost ��;W�E{+       ��K	@���A�K*

logging/current_cost0��;<��+       ��K	 p���A�K*

logging/current_cost�\�;��-+       ��K	頱��A�K*

logging/current_costL0�;xI�x+       ��K	Uα��A�K*

logging/current_coste>�;�o�]+       ��K	�����A�K*

logging/current_costk��;JH��+       ��K	e(���A�K*

logging/current_costN#�;\���+       ��K	9V���A�K*

logging/current_cost�9�;����+       ��K	킲��A�K*

logging/current_cost|��;��K+       ��K	�����A�K*

logging/current_cost���;z(Y�+       ��K	�޲��A�K*

logging/current_costR��;s�63+       ��K	����A�L*

logging/current_cost``�;�	+       ��K		@���A�L*

logging/current_cost+��;�_i+       ��K	�l���A�L*

logging/current_costG�;�S��+       ��K	ʡ���A�L*

logging/current_costrn�;�T�+       ��K	�ҳ��A�L*

logging/current_cost%��;l-��+       ��K	} ���A�L*

logging/current_cost���;��45+       ��K	�.���A�L*

logging/current_cost%��;XGo�+       ��K	j^���A�L*

logging/current_cost���;^�Ԓ+       ��K	�����A�L*

logging/current_cost!�;O��T+       ��K	�Ĵ��A�L*

logging/current_cost;��;�!��+       ��K	b����A�L*

logging/current_cost�E�;u�ʠ+       ��K	�)���A�L*

logging/current_cost<v�;��+       ��K	�V���A�L*

logging/current_cost	��;�؜	+       ��K	-����A�L*

logging/current_cost���;�A�+       ��K	����A�L*

logging/current_cost���;b��;+       ��K	����A�L*

logging/current_costK��;rJ�z+       ��K	|���A�L*

logging/current_cost+7�;��s�+       ��K	�@���A�L*

logging/current_cost���;���+       ��K	/o���A�L*

logging/current_cost��;f>N+       ��K	����A�L*

logging/current_cost4��;�uN+       ��K	�˶��A�L*

logging/current_costYM�;n��:+       ��K	�����A�L*

logging/current_cost�
�;�	?+       ��K	�$���A�L*

logging/current_cost���;�m��+       ��K	?U���A�L*

logging/current_cost�d�;�|1U+       ��K	ʅ���A�L*

logging/current_cost�s�;�x+       ��K	q����A�L*

logging/current_cost5G�;��_+       ��K	p���A�M*

logging/current_cost���;�R� +       ��K	����A�M*

logging/current_cost���;{���+       ��K	�<���A�M*

logging/current_cost���;z��
+       ��K	ok���A�M*

logging/current_cost��;@Dm+       ��K	�����A�M*

logging/current_cost��;��\�+       ��K	Ǹ��A�M*

logging/current_costI��;�2e+       ��K	{����A�M*

logging/current_cost�5�; wκ+       ��K	�!���A�M*

logging/current_cost0%�;��m+       ��K	hO���A�M*

logging/current_cost<�;�D�:+       ��K	�|���A�M*

logging/current_cost.K�;���+       ��K	����A�M*

logging/current_cost~�;YoV�+       ��K	�׹��A�M*

logging/current_costy��;?x%�+       ��K	����A�M*

logging/current_cost.��;�7|)+       ��K	�6���A�M*

logging/current_cost�C�;T\�+       ��K	�c���A�M*

logging/current_cost�~�;�6�+       ��K	$����A�M*

logging/current_cost�d�;��L�+       ��K	k����A�M*

logging/current_cost��;�h��+       ��K	����A�M*

logging/current_cost7�;��|+       ��K	����A�M*

logging/current_cost��;X^��+       ��K	~E���A�M*

logging/current_costP/�;o�J+       ��K	�����A�M*

logging/current_cost��; �2+       ��K	ܾ���A�M*

logging/current_cost���;DOL�+       ��K	����A�M*

logging/current_cost"��;r��+       ��K	nD���A�M*

logging/current_cost ��;H�]+       ��K	M����A�M*

logging/current_costD��;"eq+       ��K	d����A�N*

logging/current_cost���;��+       ��K	Y����A�N*

logging/current_cost���;�G�+       ��K	�4���A�N*

logging/current_cost�K�;k�U\